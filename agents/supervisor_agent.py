import os
from typing import Dict, List, Tuple, Any, Annotated
from datetime import datetime
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, FunctionMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import Graph, MessageGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langsmith import traceable
import sys
# Import your existing agents

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils
from .agent_campaign import build_campaign_agent
from .agent_competitor_intelligence import build_wearables_competitor_agent
LANGSMITH_API_KEY = EnvUtils().get_required_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = EnvUtils().get_required_env("LANGSMITH_PROJECT")
###############################################################################
# 1. Type Definitions
###############################################################################

class AgentState(TypedDict):
    """Type definition for agent state"""
    messages: List[Any]
    next_step: str
    agent_route: str | None
    response: str | None

###############################################################################
# 2. Agent Functions
###############################################################################

def create_agent_functions():
    """Create the agent decision and execution functions"""
    
    llm = EnvUtils().get_llm()
    
    # Initialize the existing agents
    campaign_agent = build_campaign_agent()
    wearables_agent = build_wearables_competitor_agent()

    def decide_route(state: AgentState) -> AgentState:
        """Determine which agent(s) should handle the query"""
        
        # Get the last user message
        last_message = state["messages"][-1]
        if not isinstance(last_message, HumanMessage):
            state["next_step"] = "end"
            return state
            
        prompt = PromptTemplate.from_template("""You are a supervisor that decides which specialized agent should handle a user query.
            
Available Agents:
1. Campaign Analysis Agent: Handles marketing campaign data, metrics, and performance analysis
2. Wearables Intelligence Agent: Analyzes competitor products (Fitbit, Apple Watch) and social media trends

Decide which agent(s) should handle the query:
- Return 'campaign' for campaign/marketing analysis queries
- Return 'wearables' for competitor/product analysis queries
- Return 'both' if the query requires data from both agents

Your response should be just one word: 'campaign', 'wearables', or 'both'

Query: {input}""")
        
        chain = prompt | llm
        result = chain.invoke({"input": last_message.content})
        decision = result.content.lower().strip()
        
        state["agent_route"] = decision
        state["next_step"] = "execute"
        return state

    def execute_query(state: AgentState) -> AgentState:
        """Execute the query using the appropriate agent(s)"""
        route = state["agent_route"]
        query = state["messages"][-1].content
        
        if route == "campaign":
            response = campaign_agent.invoke({"input": query})
            state["response"] = response["output"]
        
        elif route == "wearables":
            response = wearables_agent.invoke({"input": query})
            state["response"] = response["output"]
        
        elif route == "both":
            # Get responses from both agents
            campaign_response = campaign_agent.invoke({"input": query})
            wearables_response = wearables_agent.invoke({"input": query})
            
            # Combine the responses
            combine_prompt = PromptTemplate.from_template("""Combine insights from both campaign data and competitor analysis into a cohesive response.
Focus on drawing connections between marketing performance and competitor activities.

Campaign Analysis: {campaign_response}

Competitor Analysis: {competitor_response}

Original Query: {query}""")
            
            chain = combine_prompt | llm
            combined_response = chain.invoke({
                "campaign_response": campaign_response["output"],
                "competitor_response": wearables_response["output"],
                "query": query
            })
            state["response"] = combined_response.content
        
        state["next_step"] = "end"
        return state

    return {
        "decide": decide_route,
        "execute": execute_query
    }

###############################################################################
# 3. Build the Supervisor Graph
###############################################################################

def build_supervisor_graph() -> MessageGraph:
    """Build the supervisor workflow graph using latest LangGraph patterns"""
    
    # Create the workflow functions
    functions = create_agent_functions()
    
    # Define the workflow
    workflow = Graph()
    
    # Add nodes
    workflow.add_node("decide", functions["decide"])
    workflow.add_node("execute", functions["execute"])
    
    # Add edges
    workflow.add_edge("decide", "execute")
    
    # Set entry point and end point
    workflow.set_entry_point("decide")
    workflow.set_finish_point("execute")
    
    # Compile the graph
    return workflow.compile()

###############################################################################
# 4. Main Supervisor Agent Class
###############################################################################

class SupervisorAgent:
    def __init__(self):
        self.graph = build_supervisor_graph()
    
    def invoke(self, query: str) -> str:
        """Process a query through the supervisor workflow"""
        try:
            # Initialize state
            state: AgentState = {
                "messages": [HumanMessage(content=query)],
                "next_step": "decide",
                "agent_route": None,
                "response": None
            }
            
            # Run the graph
            result = self.graph.invoke(state)
            
            return result["response"] if result["response"] else "No response generated"
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

###############################################################################
# 5. Example Usage
###############################################################################

if __name__ == "__main__":
    # Initialize the supervisor
    supervisor = SupervisorAgent()
    
    # Example queries
    test_queries = [
        "How are our Facebook campaigns performing this month?",  # Campaign only
        "What's the sentiment about Apple Watch's new features?", # Wearables only
        "Compare our fitness campaign performance with competitor product reception", # Both
    ]
    
    # Test each query
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        response = supervisor.invoke(query)
        print(f"Response:\n{response}\n")