from langsmith import Client
from typing import Dict, List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
import json
import asyncio
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
from langsmith import Client
from typing import Dict, List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
import json
import asyncio

# Initialize LangSmith client
client = Client()

###############################################################################
# 1. Create Test Dataset
###############################################################################

examples = [
    {
        "question": "How are our Facebook campaign CTRs this month?",
        "expected_response": {
            "type": "campaign_analysis",
            "content": "Analysis shows Facebook campaign CTR is 2.5% this month, up from 2.1% last month"
        },
        "expected_trajectory": ["decide_route", "execute_query", "campaign_agent"],
        "expected_route": "campaign"
    },
    {
        "question": "What's the latest feedback on Apple Watch Ultra battery life?",
        "expected_response": {
            "type": "competitor_analysis",
            "content": "Recent reviews indicate Apple Watch Ultra battery life averages 36 hours"
        },
        "expected_trajectory": ["decide_route", "execute_query", "wearables_agent"],
        "expected_route": "wearables"
    },
    {
        "question": "How do our fitness campaign results compare to Fitbit's market reception?",
        "expected_response": {
            "type": "combined_analysis",
            "content": "Our fitness campaigns show 3.2% CTR while Fitbit's new products receive 85% positive sentiment"
        },
        "expected_trajectory": ["decide_route", "execute_query", "campaign_agent", "wearables_agent"],
        "expected_route": "both"
    }
]

# Create dataset in LangSmith
dataset_name = "Supervisor Agent Comprehensive Evaluation"

###############################################################################
# 2. Final Response Evaluator
###############################################################################

async def evaluate_final_response(inputs: Dict, outputs: Dict, reference: Dict) -> bool:
    """Basic evaluation of whether response type matches expected type"""
    try:
        response = json.loads(outputs["response"])
        return response["type"] == reference["expected_response"]["type"]
    except:
        return False

###############################################################################
# 3. LLM-as-Judge Evaluator
###############################################################################

class Grade(TypedDict):
    """Grading schema for LLM evaluator"""
    reasoning: Annotated[str, "Explanation of the grading decision"]
    is_correct: Annotated[bool, "Whether the response is correct"]

# LLM-as-judge instructions
GRADER_INSTRUCTIONS = """You are evaluating a supervisor agent's response quality.
Grade the response based on:
1. Correct routing to appropriate agent(s)
2. Relevant information in the response
3. Accuracy of the analysis

Return True only if all criteria are met."""

async def llm_judge_evaluation(inputs: Dict, outputs: Dict, reference: Dict) -> bool:
    """Use LLM to evaluate response quality"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    prompt = PromptTemplate.from_template("""
    Question: {question}
    Expected Response: {expected}
    Actual Response: {actual}
    
    {instructions}
    """)
    
    chain = prompt | llm
    
    result = chain.invoke({
        "question": inputs["question"],
        "expected": json.dumps(reference["expected_response"]),
        "actual": outputs["response"],
        "instructions": GRADER_INSTRUCTIONS
    })
    
    return "true" in result.content.lower()

###############################################################################
# 4. Trajectory Evaluator
###############################################################################

def evaluate_trajectory(outputs: Dict, reference: Dict) -> float:
    """Calculate what percentage of expected steps were completed"""
    try:
        actual_trajectory = outputs["trajectory"]
        expected_trajectory = reference["expected_trajectory"]
        
        # Check for subsequence match
        i = j = 0
        matches = 0
        while i < len(expected_trajectory) and j < len(actual_trajectory):
            if expected_trajectory[i] == actual_trajectory[j]:
                matches += 1
                i += 1
            j += 1
            
        return matches / len(expected_trajectory)
    except:
        return 0.0

###############################################################################
# 5. Single Step (Routing) Evaluator
###############################################################################

def evaluate_routing_step(outputs: Dict, reference: Dict) -> bool:
    """Evaluate just the routing decision"""
    try:
        response = json.loads(outputs["response"])
        if response["type"] == "campaign_analysis" and reference["expected_route"] == "campaign":
            return True
        if response["type"] == "competitor_analysis" and reference["expected_route"] == "wearables":
            return True
        if response["type"] == "combined_analysis" and reference["expected_route"] == "both":
            return True
        return False
    except:
        return False

###############################################################################
# 6. Target Function
###############################################################################

async def run_supervisor_with_tracking(inputs: Dict) -> Dict:
    """Run supervisor agent and track trajectory"""
    from supervisor_agent import SupervisorAgent
    
    supervisor = SupervisorAgent()
    trajectory = []
    
    # Track trajectory through the graph
    for event in supervisor.graph.stream(
        {"messages": [{"content": inputs["question"]}]}, 
        stream_mode="debug"
    ):
        if event["type"] == "task":
            trajectory.append(event["payload"]["name"])
    
    result = supervisor.invoke(inputs["question"])
    
    return {
        "response": result,
        "trajectory": trajectory
    }

###############################################################################
# 7. Run Evaluations
###############################################################################

def run_comprehensive_evaluation():
    """Run all evaluations"""
    
    # Create dataset if it doesn't exist
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[{"question": ex["question"]} for ex in examples],
            outputs=[{
                "expected_response": ex["expected_response"],
                "expected_trajectory": ex["expected_trajectory"],
                "expected_route": ex["expected_route"]
            } for ex in examples],
            dataset_id=dataset.id
        )
    
    # Run evaluations
    evaluations = [
        ("response_accuracy", evaluate_final_response),
        ("routing", evaluate_routing_step),
        ("trajectory", evaluate_trajectory)
    ]
    
    results = {}
    for name, evaluator in evaluations:
        eval_results = client.evaluate(
            run_supervisor_with_tracking,
            data=dataset_name,
            evaluation_name=name,
            evaluators=[evaluator],
            experiment_prefix="supervisor-agent",
            input_mapper=lambda x: {"question": x["question"]},
        )
        results[name] = eval_results.to_pandas()
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_evaluation()
    
    print("\nEvaluation Results:")
    for name, df in results.items():
        print(f"\n{name.upper()} Evaluation:")
        print(df)

if __name__ == "__main__":
    results = asyncio.run(run_comprehensive_evaluation())
    
    print("\nEvaluation Results:")
    print("\n1. Final Response Evaluation:")
    print(results["final_response"])
    
    print("\n2. LLM Judge Evaluation:")
    print(results["llm_judge"])
    
    print("\n3. Trajectory Evaluation:")
    print(results["trajectory"])
    
    print("\n4. Routing Evaluation:")
    print(results["routing"])