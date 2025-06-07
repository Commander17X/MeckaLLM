from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn

from ..core.model import MeckaLLMModel
from ..distributed.coordinator import TaskCoordinator

app = FastAPI(title="MeckaLLM API")
model = MeckaLLMModel()
coordinator = TaskCoordinator()

class TaskRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    num_return_sequences: Optional[int] = 1

class TaskResponse(BaseModel):
    task_id: str
    status: str

@app.post("/generate", response_model=TaskResponse)
async def generate_text(request: TaskRequest):
    """
    Submit a text generation task.
    """
    try:
        task_id = coordinator.submit_task(
            "text_generation",
            {
                "prompt": request.prompt,
                "max_length": request.max_length,
                "temperature": request.temperature,
                "num_return_sequences": request.num_return_sequences
            }
        )
        return TaskResponse(task_id=task_id, status="pending")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of a task.
    """
    status = coordinator.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 