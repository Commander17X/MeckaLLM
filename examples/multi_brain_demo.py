import asyncio
import time
from meckallm.brain.task_coordinator import TaskCoordinator, TaskPriority
from meckallm.brain.quantum_brain import Task

async def main():
    # Initialize coordinator
    coordinator = TaskCoordinator()
    
    # Create example tasks
    tasks = [
        # High priority tasks
        Task(
            type="browser",
            description="Research quantum computing",
            parameters={
                "url": "https://arxiv.org/list/quant-ph/recent",
                "actions": [
                    {"type": "scroll", "amount": 5},
                    {"type": "click", "x": 500, "y": 300}
                ]
            },
            priority=TaskPriority.HIGH
        ),
        Task(
            type="code",
            description="Implement quantum algorithm",
            parameters={
                "language": "python",
                "code": "def quantum_algorithm():\n    pass"
            },
            priority=TaskPriority.HIGH
        ),
        
        # Medium priority tasks
        Task(
            type="music",
            description="Play background music",
            parameters={
                "action": "play",
                "playlist": "Focus"
            },
            priority=TaskPriority.MEDIUM
        ),
        Task(
            type="email",
            description="Check emails",
            parameters={
                "action": "check",
                "folder": "inbox"
            },
            priority=TaskPriority.MEDIUM
        ),
        
        # Background tasks
        Task(
            type="monitor",
            description="Monitor system resources",
            parameters={
                "metrics": ["cpu", "memory", "gpu"]
            },
            priority=TaskPriority.BACKGROUND
        ),
        Task(
            type="backup",
            description="Backup important files",
            parameters={
                "paths": ["/important/docs", "/projects"]
            },
            priority=TaskPriority.BACKGROUND
        )
    ]
    
    # Add tasks to coordinator
    task_ids = []
    for task in tasks:
        task_id = await coordinator.add_task(task)
        task_ids.append(task_id)
        print(f"Added task: {task.description} (ID: {task_id})")
        
    # Start load monitoring
    monitor_task = asyncio.create_task(coordinator.monitor_loads())
    
    # Wait for tasks to complete
    while True:
        # Get system status
        status = coordinator.get_system_status()
        print("\nSystem Status:")
        print(f"Active Tasks: {status['active_tasks']}")
        print(f"Total Completed: {status['total_completed']}")
        
        # Print screen statuses
        print("\nScreen Statuses:")
        for screen_id, screen_status in status['screen_statuses'].items():
            print(f"Screen {screen_id}:")
            print(f"  Active Tasks: {screen_status['active_tasks']}")
            print(f"  Queue Length: {screen_status['queue_length']}")
            print(f"  Brain Load: {screen_status['brain_load']}")
            
        # Check if all tasks are done
        if status['active_tasks'] == 0:
            break
            
        # Wait before next update
        await asyncio.sleep(5)
        
    # Stop monitoring
    monitor_task.cancel()
    
    # Save state
    coordinator.save_state("brain_state.json")
    print("\nState saved to brain_state.json")

if __name__ == "__main__":
    asyncio.run(main()) 