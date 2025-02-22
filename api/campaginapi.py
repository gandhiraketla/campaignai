from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import sys
import os
from fastapi.middleware.cors import CORSMiddleware


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from util.envutils import EnvUtils


# Import the SupervisorAgent
from agents.supervisor_agent import SupervisorAgent

# Create FastAPI app
app = FastAPI(
    title="Supervisor Agent API",
    description="API for querying multiple specialized agents",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173"  # React development server
          # Add your production domain when deployed
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request model
class QueryRequest(BaseModel):
    query: str

# Response model
class QueryResponse(BaseModel):
    response: str

# Initialize the supervisor agent
supervisor = SupervisorAgent()

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query through the supervisor agent
    
    - Automatically routes the query to appropriate specialized agents
    - Supports campaign, wearables, or combined queries
    """
    try:
        # Invoke the supervisor agent with the query
        response = supervisor.invoke(request.query)
        return QueryResponse(response=response)
    
    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

# Run the application (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)