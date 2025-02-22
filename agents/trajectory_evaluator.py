from langsmith import Client
from typing import Dict, List
from supervisor_agent import SupervisorAgent
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils
LANGSMITH_API_KEY = EnvUtils().get_required_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = EnvUtils().get_required_env("LANGSMITH_PROJECT")
import logging
from typing import Dict, List
from langsmith import Client
from langchain_core.messages import HumanMessage
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("TrajectoryEvaluator")

# Initialize LangSmith client
client = Client()

# Test dataset with expected trajectories
examples = [
    {
        "question": "How are our Facebook campaign CTRs this month?",
        "expected_trajectory": ["decide", "execute", "campaign_agent"],
        "expected_response": {
            "type": "campaign_analysis",
            "metrics": ["CTR", "impressions", "clicks"]
        }
    },
    {
        "question": "What's the latest sentiment about Apple Watch features?",
        "expected_trajectory": ["decide", "execute", "wearables_agent"],
        "expected_response": {
            "type": "competitor_analysis",
            "focus": "sentiment"
        }
    },
    {
        "question": "How do our fitness campaign results compare to Fitbit's market reception?",
        "expected_trajectory": ["decide", "execute", "campaign_agent", "wearables_agent"],
        "expected_response": {
            "type": "combined_analysis",
            "components": ["campaign_metrics", "competitor_sentiment"]
        }
    }
]

def create_evaluation_dataset():
    """Create or get the evaluation dataset"""
    dataset_name = "Supervisor Agent Trajectory Analysis"
    
    if not client.has_dataset(dataset_name=dataset_name):
        logger.info(f"Creating new dataset: {dataset_name}")
        dataset = client.create_dataset(dataset_name=dataset_name)
        
        # Create examples
        client.create_examples(
            inputs=[{"question": ex["question"]} for ex in examples],
            outputs=[{
                "expected_trajectory": ex["expected_trajectory"],
                "expected_response": ex["expected_response"]
            } for ex in examples],
            dataset_id=dataset.id
        )
        logger.info(f"Created {len(examples)} examples in dataset")
    else:
        logger.info(f"Using existing dataset: {dataset_name}")
    
    return dataset_name

def evaluate_trajectory(run) -> Dict:
    """
    Evaluate the agent's trajectory against expected path.
    Args:
        run: RunTree object containing the run information
    Returns:
        Dict containing score and analysis
    """
    try:
        # Extract trajectories
        actual_trajectory = run.outputs.get("trajectory", []) if hasattr(run.outputs, "get") else []
        expected_trajectory = run.reference.get("expected_trajectory", [])
        
        if not expected_trajectory:
            return {
                "score": 0.0,
                "reasoning": "No expected trajectory provided"
            }

        # Calculate matches while preserving order
        matches = 0
        position = 0
        expected_pos = {}
        
        # Create position map for expected steps
        for i, step in enumerate(expected_trajectory):
            expected_pos[step] = i
        
        # Check matches in order
        for step in actual_trajectory:
            if step in expected_pos and expected_pos[step] >= position:
                matches += 1
                position = expected_pos[step] + 1
        
        # Calculate score
        score = matches / len(expected_trajectory)
        
        # Generate analysis
        analysis = []
        analysis.append(f"Matched {matches} out of {len(expected_trajectory)} expected steps")
        
        # Check for missing or extra steps
        missing = set(expected_trajectory) - set(actual_trajectory)
        extra = set(actual_trajectory) - set(expected_trajectory)
        
        if missing:
            analysis.append(f"Missing required steps: {', '.join(missing)}")
        if extra:
            analysis.append(f"Extra steps taken: {', '.join(extra)}")
            
        # Check step order
        if matches > 0:
            actual_order = [step for step in actual_trajectory if step in expected_trajectory]
            if actual_order != expected_trajectory:
                analysis.append("Steps were not executed in the expected order")
        
        return {
            "score": float(score),
            "reasoning": "\n".join(analysis)
        }
        
    except Exception as e:
        logger.error(f"Error in trajectory evaluation: {e}")
        return {
            "score": 0.0,
            "reasoning": f"Evaluation error: {str(e)}"
        }

def run_agent_with_tracking(inputs: Dict) -> Dict:
    """
    Run the supervisor agent while tracking its execution path
    Args:
        inputs: Dict containing the input question
    Returns:
        Dict containing trajectory and response
    """
    from supervisor_agent import SupervisorAgent
    
    try:
        supervisor = SupervisorAgent()
        trajectory = []
        
        # Initialize state
        state = {
            "messages": [HumanMessage(content=inputs["question"])],
            "next_step": "decide",
            "trajectory": trajectory
        }
        
        # Run agent with tracing
        for event in supervisor.graph.stream(
            state,
            stream_mode="debug"
        ):
            if event["type"] == "start":
                trajectory.append(event["node"]["name"])
            elif event["type"] == "tool" and "name" in event:
                trajectory.append(event["name"])
        
        # Get final response
        result = supervisor.invoke(inputs["question"])
        
        return {
            "trajectory": trajectory,
            "response": result
        }
        
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        return {
            "trajectory": trajectory,
            "error": str(e)
        }

def main():
    """Run the trajectory evaluation"""
    try:
        # Create or get dataset
        dataset_name = create_evaluation_dataset()
        
        # Run evaluation
        logger.info("Starting trajectory evaluation")
        results = client.evaluate(
            run_agent_with_tracking,
            dataset_name,
            evaluators=[evaluate_trajectory],
            tags=["trajectory_analysis"]
        )
        
        # Display results
        df = results.to_pandas()
        print("\nTrajectory Evaluation Results:")
        
        for _, row in df.iterrows():
            print(f"\nQuestion: {row['inputs']['question']}")
            
            if 'results' in row and isinstance(row['results'], dict):
                print(f"Score: {row['results'].get('score', 'N/A')}")
                print(f"Analysis:\n{row['results'].get('reasoning', 'No analysis available')}")
                
                # Show actual trajectory if available
                if 'outputs' in row and isinstance(row['outputs'], dict):
                    trajectory = row['outputs'].get('trajectory', [])
                    print(f"Actual trajectory: {' -> '.join(trajectory)}")
            else:
                print("No evaluation results available")
                
        logger.info("Evaluation complete")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        raise

if __name__ == "__main__":
    main()