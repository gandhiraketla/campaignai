import os
import sys
import re
import logging
import mysql.connector
from mysql.connector import Error
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import Graph, MessageGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langsmith import traceable

# Import parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils
LANGSMITH_API_KEY = EnvUtils().get_required_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = EnvUtils().get_required_env("LANGSMITH_PROJECT")
###############################################################################
# 1. Logging & Env Setup
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("AgentCampaign")

OPENAI_API_KEY = EnvUtils().get_required_env("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set! The LLM or embeddings may fail.")
 
DB_HOST = EnvUtils().get_required_env("DB_HOST")
DB_USER = EnvUtils().get_required_env("DB_USER")
DB_PASS = EnvUtils().get_required_env("DB_PASSWORD")
DB_NAME = EnvUtils().get_required_env("DB_NAME")
CHROMA_PERSIST_DIR = EnvUtils().get_required_env("CHROMA_PERSIST_DIR")

def get_llm():
    """Get the appropriate LLM based on environment configuration"""
    model_type = EnvUtils().get_required_env("MODEL_TYPE").lower()
    if model_type == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="mistral",
            temperature=0
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )

###############################################################################
# 2. LLM-based SQL Generation
###############################################################################

SCHEMA_DESCRIPTION = """
You have a table named 'campaigns' with columns:
- id (INT, PRIMARY KEY)
- campaign_name (VARCHAR)
- channel (VARCHAR)                  -- e.g., Facebook, Google, LinkedIn
- campaign_type ENUM('awareness', 'conversion')
- start_date DATE
- end_date DATE
- status ENUM('active', 'completed')
- budget DECIMAL(10,2)              -- Total allocated budget
- spend DECIMAL(10,2)               -- Actual spend
- impressions INT
- clicks INT
- conversions INT
- revenue DECIMAL(10,2)             -- Revenue generated
- notes TEXT
- created_at TIMESTAMP

Your job: convert the user's question into a valid MySQL SELECT statement.
Constraints:
1) Return only a SELECT query, no additional commentary.
2) If user references columns not in the schema, do your best with existing columns.
3) The user might reference partial or approximate column names—map them to the actual columns if possible.
4) If the user asks for specific fields (e.g., "How many clicks for Campaign X?"), only retrieve those specific fields.
5) For performance-related queries (comparing campaigns, optimization, performance analysis), always include:
   - CTR (Click-Through Rate): (clicks / impressions) * 100
   - Cost per Conversion: spend / conversions
   - ROAS (Return on Ad Spend): revenue / spend
   - Conversion Rate: (conversions / clicks) * 100
6) Keep queries structured and efficient, selecting only relevant columns.
7) Output must be purely SQL, no explanation.

user's question:
{input}
"""

@traceable(name="generate_sql_from_llm")

def generate_sql_from_llm(user_query: str) -> str:
    """
    Asks an LLM to produce a SELECT statement for the 'campaigns' table 
    based on the user's natural language question.
    WARNING: In production, parse or validate the output to ensure only read-only queries.
    """
    query_prompt = PromptTemplate.from_template(SCHEMA_DESCRIPTION)
    llm = get_llm()
    evaluation_chain = query_prompt | llm
    eval_input = {
        "input": user_query
    }
    evaluation_result = evaluation_chain.invoke(eval_input)
    sql_candidate = evaluation_result.content.strip()
    
    # Basic sanity check: Must contain "SELECT"
    if not re.search(r"select", sql_candidate.lower()):
        logger.warning("LLM returned a suspicious query without SELECT: %s", sql_candidate)
    return sql_candidate

###############################################################################
# 3. Execute the LLM-generated SQL in MySQL
###############################################################################

@traceable(name="execute_llm_sql")
def execute_llm_sql(sql_query: str):
    """
    Safely run the LLM-generated SQL in MySQL.
    Returns a list of row dicts if successful, or raises an exception.
    """
    conn = None
    try:
        conn = mysql.connector.connect(
            host=DB_HOST, 
            user=DB_USER, 
            password=DB_PASS, 
            database=DB_NAME
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        cursor.close()
        logger.info("Executed SQL: '%s' -> rowcount=%d", sql_query, len(rows))
        return rows
    except Error as e:
        logger.exception("DB error with LLM-generated SQL.")
        raise e
    finally:
        if conn and conn.is_connected():
            conn.close()

@traceable(name="dynamic_campaign_sql_tool_func")
def dynamic_campaign_sql_tool_func(user_query: str) -> str:
    """
    1. Generate SQL via LLM
    2. Execute the SQL
    3. Summarize results
    """
    sql_stmt = generate_sql_from_llm(user_query)
    try:
        rows = execute_llm_sql(sql_stmt)
    except Exception as exc:
        logger.error("Error executing LLM SQL: %s", exc)
        return f"Error executing query: {exc}"

    if not rows:
        return "No results found."

    # Summarize up to 3 rows
    lines = []
    for r in rows[:3]:
        c_name = r.get("campaign_name", "N/A")
        channel = r.get("channel", "N/A")
        spend = r.get("spend", 0)
        revenue = r.get("revenue", 0)
        lines.append(
            f"- {c_name} on {channel}, spend=${spend}, revenue=${revenue}"
        )
    return "LLM-based SQL Results:\n" + "\n".join(lines)

###############################################################################
# 4. Local Chroma Doc Retrieval
###############################################################################

@traceable(name="retrieve_docs_func")
def retrieve_docs_func(user_query: str) -> str:
    """
    Does a similarity search on local Chroma index for marketing docs.
    """
    try:
        embeddings = OpenAIEmbeddings()
        vs = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
        docs = vs.similarity_search(user_query, k=2)
        if not docs:
            return "No relevant doc snippets found."
        snippet = []
        for d in docs:
            src = d.metadata.get("source", "Unknown source")
            snippet.append(f"[{src}]\n{d.page_content[:300]}...")
        return "\n---\n".join(snippet)

    except Exception as e:
        logger.exception("Error in doc retrieval.")
        return f"Doc retrieval failed: {e}"

###############################################################################
# 5. Build the Agent with Two Tools
###############################################################################

@traceable(name="build_campaign_agent")
def build_campaign_agent():
    """
    Creates an Agent with:
      1) dynamic_campaign_sql_tool: queries MySQL via LLM-generated SQL
      2) docs_retrieval_tool: local doc retrieval from Chroma
    """
    
    llm = get_llm()
    
    # Define the tools
    dynamic_campaign_sql_tool = Tool(
        name="dynamic_campaign_sql_tool",
        func=dynamic_campaign_sql_tool_func,
        description=(
            "Generate an SQL SELECT query for the 'campaigns' table "
            "from user instructions using an LLM, then execute it and summarize results."
        ),
    )

    docs_retrieval_tool = Tool(
        name="docs_retrieval_tool",
        func=retrieve_docs_func,
        description=(
            "Retrieve relevant doc snippets from a local Chroma store "
            "based on user's question about marketing/brand guidelines."
        ),
    )

    tools = [dynamic_campaign_sql_tool, docs_retrieval_tool]

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that helps with marketing campaigns analysis. You have access to campaign data and documentation."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False
    )

    logger.info("Campaign Agent built with 2 tools (dynamic SQL + doc retrieval).")
    return agent_executor

###############################################################################
# 6. Main (Demo)
###############################################################################

if __name__ == "__main__":
    logger.info("Starting the Campaign Agent Demo...")

    agent_executor = build_campaign_agent()

    # Example queries:
    queries = [
        "Compare the campaign peformance between Facebook and LinkedIn this year",
    ]

    for q in queries:
        logger.info("USER: %s", q)
        try:
            response = agent_executor.invoke({"input": q})
            print(f"\nUSER: {q}\nAGENT:\n{response['output']}\n---\n")
        except Exception as exc:
            logger.exception("Agent failed on query='%s'", q)
            print(f"Error: {exc}")