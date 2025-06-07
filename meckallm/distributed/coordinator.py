from typing import Dict, List, Optional
import redis
import json
from datetime import datetime

class TaskCoordinator:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize the task coordinator.
        
        Args:
            redis_url: URL for Redis connection
        """
        self.redis_client = redis.from_url(redis_url)
        
    def submit_task(self, task_type: str, parameters: Dict) -> str:
        """
        Submit a new task to the system.
        
        Args:
            task_type: Type of task to perform
            parameters: Task parameters
            
        Returns:
            Task ID
        """
        task_id = f"task_{datetime.now().timestamp()}"
        task_data = {
            "type": task_type,
            "parameters": parameters,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        self.redis_client.hset(f"task:{task_id}", mapping=task_data)
        self.redis_client.lpush("task_queue", task_id)
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict:
        """
        Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status information
        """
        task_data = self.redis_client.hgetall(f"task:{task_id}")
        return task_data if task_data else {}
    
    def update_task_status(self, task_id: str, status: str, result: Optional[Dict] = None):
        """
        Update the status of a task.
        
        Args:
            task_id: ID of the task
            status: New status
            result: Task result (optional)
        """
        updates = {"status": status, "updated_at": datetime.now().isoformat()}
        if result:
            updates["result"] = json.dumps(result)
            
        self.redis_client.hset(f"task:{task_id}", mapping=updates)
    
    def get_pending_tasks(self) -> List[str]:
        """
        Get list of pending tasks.
        
        Returns:
            List of task IDs
        """
        return self.redis_client.lrange("task_queue", 0, -1) 