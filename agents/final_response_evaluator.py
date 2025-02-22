import sys
import os
from langchain.prompts import PromptTemplate

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
from typing import Dict, TypedDict, Annotated
from langchain_openai import ChatOpenAI
import json
# Initialize LangSmith client
client = Client()
# Create test dataset
examples = [
    {
        "question": "How are our Facebook campaign CTRs this month?",
        "response": {
            "type": "campaign_analysis",
            "content": "The average Click-Through Rate (CTR) for our Facebook campaigns this month is 3.10%"
        }
    },
    {
        "question": "What's the ROI on our Google Ads campaigns?",
        "response": {
            "type": "campaign_analysis",
            "content": "The ROI on our Google Ads campaigns is 65%. This means that for every dollar spent on Google Ads, there was a return of $1.65"
        }
    },
    {
        "question": "Which campaign had the highest conversion rate?",
        "response": {
            "type": "campaign_analysis",
            "content": "The campaign with the highest conversion rate is the 'Google Awareness Campaign 329' with a conversion rate of 9.98%"
        }
    }
]

# Dataset setup
dataset_name = "Campaign Analysis Agent: Final Response"

if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
        inputs=[{"question": ex["question"]} for ex in examples],
        outputs=[{"response": ex["response"]} for ex in examples],
        dataset_id=dataset.id
    )

# Grader instructions
GRADER_INSTRUCTIONS = """You are evaluating responses from a marketing campaign analysis agent.

You will be given:
QUESTION: The original query about campaign performance
GROUND TRUTH: The expected response with accurate campaign metrics
ACTUAL RESPONSE: The agent's response to evaluate

Grade based on these criteria:
1. Factual Accuracy: Numbers and metrics must match the ground truth
2. Completeness: All relevant metrics from the ground truth are included
3. Relevance: Response directly answers the original question

Score as True only if all criteria are met.
Explain your reasoning step by step."""

# Output schema for the grader
class Grade(TypedDict):
    """Evaluation result for campaign analysis responses"""
    reasoning: Annotated[str, "Detailed explanation of the grading decision"]
    is_correct: Annotated[bool, "True if response meets all criteria, False otherwise"]

def evaluate_final_response(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Evaluate if the agent's response matches the reference response"""
    
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    evaluation_prompt = """You are evaluating responses from a marketing campaign analysis agent.

Given:
QUESTION: {question}
GROUND TRUTH: {ground_truth}
ACTUAL RESPONSE: {actual}

Evaluate based on these criteria:
1. Factual Accuracy: Numbers and metrics must match the ground truth
2. Completeness: All relevant metrics are included
3. Relevance: Response directly answers the original question

Output your evaluation as a single number between 0 and 1, where:
0.0 = Completely incorrect or irrelevant
0.5 = Partially correct or missing some information
1.0 = Completely correct and complete

Just respond with a single number and a brief explanation."""
    
    formatted_prompt = evaluation_prompt.format(
        question=inputs["question"],
        ground_truth=reference_outputs["response"],
        actual=outputs["response"]
    )
    
    messages = [{"role": "system", "content": formatted_prompt}]
    result = llm.invoke(messages)
    
    try:
        # Parse the result which should be a number followed by explanation
        response_text = result.content
        score = float(response_text.split()[0])  # Get first word and convert to float
        reasoning = " ".join(response_text.split()[1:])  # Rest is the explanation
        
        return {
            "score": score,
            "reasoning": reasoning
        }
    except Exception as e:
        print(f"Error parsing evaluation result: {e}")
        print(f"Raw response: {response_text}")
        return {
            "score": 0.0,
            "reasoning": f"Error in evaluation process: {str(e)}"
        }

def run_supervisor(inputs: dict) -> dict:
    """Run the supervisor agent and return response"""
    from supervisor_agent import SupervisorAgent
    
    supervisor = SupervisorAgent()
    result = supervisor.invoke(inputs["question"])
    
    return {"response": result}

if __name__ == "__main__":
    # Run evaluation
    evaluation_results = client.evaluate(
        run_supervisor,
        data=dataset_name,
        evaluators=[evaluate_final_response],
        experiment_prefix="supervisor-agent-final-response"
    )
    
    # Show results
    results_df = evaluation_results.to_pandas()
    print("\nEvaluation Results:")
    print(results_df)