import os
import sys
import re
import logging
import mysql.connector

from typing import Dict, List, TypedDict, Annotated
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
You are an expert SQL developer specializing in marketing analytics. Using the campaigns table described below, generate precise MySQL queries that calculate marketing metrics.

Table: campaigns
Columns:
- id (INT, PRIMARY KEY)
- campaign_name (VARCHAR)
- channel (VARCHAR)                  -- Valid values: Facebook, Google, LinkedIn only
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

Required Marketing Metrics (Always calculate when relevant):
1. CTR = (clicks / impressions) * 100
2. Cost per Conversion = spend / conversions
3. ROAS = revenue / spend
4. Conversion Rate = (conversions / clicks) * 100

Query Requirements:
1. ALWAYS use NULLIF() when dividing to avoid division by zero errors
   Example: (clicks / NULLIF(impressions, 0)) * 100 as ctr
2. ROUND all calculated metrics to 2 decimal places
   Example: ROUND((clicks / NULLIF(impressions, 0)) * 100, 2) as ctr
3. Include relevant grouping and filters based on channel, date ranges, or status
4. Select only the columns needed to answer the question
5. For date ranges, use proper DATE() functions
6. Always alias calculated columns with clear names

Your task: Generate a MySQL SELECT statement that answers this question:
{input}

Return only the SQL query without any explanation or commentary.
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
def summarize_sql_results(rows: List[Dict], max_rows: int = 5) -> str:
    """Summarize SQL results to prevent token overflow"""
    if not rows:
        return "No results found."
        
    total_rows = len(rows)
    summarized_rows = rows[:max_rows]
    
    # Calculate aggregate metrics if present
    metrics = {}
    numeric_columns = ['impressions', 'clicks', 'conversions', 'spend', 'revenue', 'ctr', 'conversion_rate', 'roas']
    
    for col in numeric_columns:
        values = [float(row[col]) for row in rows if col in row and row[col] is not None]
        if values:
            metrics[col] = {
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
    
    # Format summary
    summary = []
    summary.append(f"Total results: {total_rows} (showing first {len(summarized_rows)})")
    
    # Add metric summaries if available
    for col, stats in metrics.items():
        summary.append(f"{col.upper()}: Avg={stats['avg']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}")
    
    # Add sample rows
    summary.append("\nSample results:")
    for row in summarized_rows:
        row_items = [f"{k}={v}" for k, v in row.items()]
        summary.append("  " + ", ".join(row_items))
    
    return "\n".join(summary)
@traceable(name="dynamic_campaign_sql_tool_func")
def dynamic_campaign_sql_tool_func(user_query: str) -> str:
    """
    1. Generate SQL via LLM
    2. Execute the SQL
    3. Use LLM to format results based on the query context
    """
    RESULT_FORMATTER_PROMPT = """You are an expert at interpreting SQL results for marketing campaigns.
Given the original user question and SQL results, create a clear and concise summary.

User Question: {user_query}

SQL Query Used: {sql_query}

SQL Results:
{sql_results}

Format your response considering:
1. If user asked about metrics (CTR, ROAS, etc.), highlight those numbers
2. If user asked for comparisons, show the differences
3. If user asked for trends, summarize the pattern
4. Include relevant context from the data
5. Format numbers properly (e.g., percentages with 2 decimals, large numbers with commas)

Return a concise, natural language response that directly answers the user's question."""

    # Generate and execute SQL
    sql_stmt = generate_sql_from_llm(user_query)
    try:
        rows = execute_llm_sql(sql_stmt)
    except Exception as exc:
        logger.error("Error executing LLM SQL: %s", exc)
        return f"Error executing query: {exc}"
    results_summary = summarize_sql_results(rows)
    if not rows:
        return "No results found."

    # Convert SQL results to readable format
    results_text = ""
    for row in rows:
        row_items = []
        for key, value in row.items():
            if isinstance(value, (int, float)):
                # Format numbers with commas and 2 decimal places if needed
                row_items.append(f"{key}: {value:,.2f}")
            else:
                row_items.append(f"{key}: {value}")
        results_text += "\n" + ", ".join(row_items)

    # Use LLM to format results
    llm = EnvUtils().get_llm()
    format_prompt = PromptTemplate.from_template(RESULT_FORMATTER_PROMPT)
    
    chain = format_prompt | llm
    
    formatted_result = chain.invoke({
        "user_query": user_query,
        "sql_query": sql_stmt,
        "sql_results": results_summary
    })

    return formatted_result.content

# Example usage:
# user_query = "What's our Facebook campaign CTR this month?"
# This would return something like:
# "Your Facebook campaigns this month have achieved an average CTR of 2.45%, 
#  with the best performing campaign 'Summer Sale' reaching 3.12% CTR. 
#  This represents a 0.5% improvement over last month's average."

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
        "Which campaign had the highest conversion rate?"
    ]

    for q in queries:
        logger.info("USER: %s", q)
        try:
            response = agent_executor.invoke({"input": q})
            print(f"\nUSER: {q}\nAGENT:\n{response['output']}\n---\n")
        except Exception as exc:
            logger.exception("Agent failed on query='%s'", q)
            print(f"Error: {exc}")