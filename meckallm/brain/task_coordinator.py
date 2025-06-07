import asyncio
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import time
from collections import defaultdict
import numpy as np
import torch
from pathlib import Path
import json

from meckallm.brain.quantum_brain import BrainCluster, BrainConfig, Task
from meckallm.interface.system_controller import SystemController

@dataclass
class TaskPriority:
    CRITICAL: int = 0
    HIGH: int = 1
    MEDIUM: int = 2
    LOW: int = 3
    BACKGROUND: int = 4

class TaskCoordinator:
    def __init__(self, config_path: Optional[str] = None):
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize system controller
        self.system = SystemController()
        
        # Initialize brain cluster
        self.brain_config = BrainConfig()
        self.brain_cluster = BrainCluster(self.brain_config)
        
        # Initialize task tracking
        self.active_tasks: Dict[str, Task] = {}
        self.task_history: List[Dict[str, Any]] = []
        self.screen_tasks: Dict[int, List[str]] = defaultdict(list)
        
        # Load configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize task queues
        self.task_queues: Dict[int, List[Task]] = defaultdict(list)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
            
    async def add_task(self, task: Task, screen_id: Optional[int] = None) -> str:
        """Add a new task to the system"""
        # Generate task ID
        task_id = f"task_{len(self.active_tasks)}_{int(time.time())}"
        
        # Assign screen if not specified
        if screen_id is None:
            screen_id = self._select_screen(task)
            
        # Add to active tasks
        self.active_tasks[task_id] = task
        self.screen_tasks[screen_id].append(task_id)
        
        # Add to queue
        self.task_queues[screen_id].append(task)
        
        # Process task
        asyncio.create_task(self._process_task(task_id, screen_id))
        
        return task_id
        
    def _select_screen(self, task: Task) -> int:
        """Select best screen for task"""
        # Get current screen loads
        screen_loads = self.brain_cluster.get_screen_loads()
        
        # Consider task priority
        if task.priority <= TaskPriority.HIGH:
            # High priority tasks go to least loaded screen
            return min(screen_loads.items(), key=lambda x: x[1])[0]
        else:
            # Other tasks can be distributed more evenly
            return np.random.choice(list(screen_loads.keys()))
            
    async def _process_task(self, task_id: str, screen_id: int):
        """Process a task using the brain cluster"""
        task = self.active_tasks[task_id]
        
        try:
            # Process through brain cluster
            result = await self.brain_cluster.process_task(
                {
                    'type': task.type,
                    'data': task.parameters,
                    'priority': task.priority
                },
                screen_id
            )
            
            # Update task status
            task.status = "completed"
            task.result = result
            
            # Add to history
            self.task_history.append({
                'id': task_id,
                'type': task.type,
                'status': task.status,
                'result': result,
                'screen_id': screen_id,
                'completion_time': time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {e}")
            task.status = "failed"
            
        finally:
            # Clean up
            self.screen_tasks[screen_id].remove(task_id)
            del self.active_tasks[task_id]
            
    async def monitor_loads(self):
        """Monitor and rebalance loads across screens"""
        while True:
            try:
                # Check screen loads
                screen_loads = self.brain_cluster.get_screen_loads()
                
                # Rebalance if needed
                if max(screen_loads.values()) > min(screen_loads.values()) * 1.5:
                    self.brain_cluster.rebalance_loads()
                    
                # Wait before next check
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in load monitoring: {e}")
                await asyncio.sleep(1)
                
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'id': task_id,
                'type': task.type,
                'status': task.status,
                'priority': task.priority
            }
        return None
        
    def get_screen_status(self, screen_id: int) -> Dict[str, Any]:
        """Get status of a specific screen"""
        return {
            'active_tasks': len(self.screen_tasks[screen_id]),
            'queue_length': len(self.task_queues[screen_id]),
            'brain_load': self.brain_cluster.get_screen_loads()[screen_id]
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'active_tasks': len(self.active_tasks),
            'screen_statuses': {
                screen_id: self.get_screen_status(screen_id)
                for screen_id in self.screen_tasks.keys()
            },
            'total_completed': len(self.task_history)
        }
        
    def save_state(self, path: str):
        """Save coordinator state"""
        state = {
            'config': self.config,
            'task_history': self.task_history,
            'screen_assignments': dict(self.screen_tasks)
        }
        
        # Save brain cluster state
        brain_state_path = Path(path).parent / 'brain_state.pt'
        self.brain_cluster.save_state(str(brain_state_path))
        
        # Save coordinator state
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, path: str):
        """Load coordinator state"""
        # Load brain cluster state
        brain_state_path = Path(path).parent / 'brain_state.pt'
        self.brain_cluster.load_state(str(brain_state_path))
        
        # Load coordinator state
        with open(path, 'r') as f:
            state = json.load(f)
            
        self.config = state['config']
        self.task_history = state['task_history']
        self.screen_tasks = defaultdict(list, state['screen_assignments']) 