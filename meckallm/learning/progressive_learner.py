import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
import psutil
import win32gui
import win32process
import win32con
import win32api
import time
import json
from pathlib import Path
import threading
import queue
import keyboard
import mouse
from PIL import ImageGrab
import cv2
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..optimization.resource_manager import ResourceManager

@dataclass
class ProgressiveLearningConfig:
    # Learning settings
    learning_rate: float = 1e-5
    batch_size: int = 4
    max_length: int = 8192
    update_interval: float = 60.0  # seconds
    
    # Resource settings
    max_memory_gb: float = 4.0
    max_disk_gb: float = 10.0
    compression_level: int = 3
    
    # Monitoring settings
    capture_screen: bool = True
    capture_keystrokes: bool = True
    capture_mouse: bool = True
    capture_apps: bool = True
    capture_system: bool = True
    
    # Storage settings
    data_dir: str = "data/progressive_learning"
    model_dir: str = "models/progressive_learning"

class ActivityMonitor:
    def __init__(self, config: ProgressiveLearningConfig):
        self.config = config
        self.activity_queue = queue.Queue()
        self.running = False
        self.last_activity = {}
        
    def start_monitoring(self):
        """Start monitoring system activities"""
        self.running = True
        
        # Start monitoring threads
        if self.config.capture_apps:
            threading.Thread(target=self._monitor_applications, daemon=True).start()
        if self.config.capture_screen:
            threading.Thread(target=self._monitor_screen, daemon=True).start()
        if self.config.capture_keystrokes:
            threading.Thread(target=self._monitor_keystrokes, daemon=True).start()
        if self.config.capture_mouse:
            threading.Thread(target=self._monitor_mouse, daemon=True).start()
        if self.config.capture_system:
            threading.Thread(target=self._monitor_system, daemon=True).start()
            
    def stop_monitoring(self):
        """Stop monitoring system activities"""
        self.running = False
        
    def _monitor_applications(self):
        """Monitor active applications"""
        while self.running:
            try:
                # Get foreground window
                hwnd = win32gui.GetForegroundWindow()
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                process = psutil.Process(pid)
                
                # Get window title
                title = win32gui.GetWindowText(hwnd)
                
                # Record activity
                activity = {
                    "type": "application",
                    "timestamp": time.time(),
                    "app_name": process.name(),
                    "window_title": title,
                    "pid": pid
                }
                
                self.activity_queue.put(activity)
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Error monitoring applications: {e}")
                
    def _monitor_screen(self):
        """Monitor screen content"""
        while self.running:
            try:
                # Capture screen
                screenshot = ImageGrab.grab()
                screenshot_np = np.array(screenshot)
                
                # Resize for efficiency
                screenshot_np = cv2.resize(screenshot_np, (320, 240))
                
                # Record activity
                activity = {
                    "type": "screen",
                    "timestamp": time.time(),
                    "image": screenshot_np
                }
                
                self.activity_queue.put(activity)
                time.sleep(5)  # Capture every 5 seconds
                
            except Exception as e:
                logging.error(f"Error monitoring screen: {e}")
                
    def _monitor_keystrokes(self):
        """Monitor keyboard input"""
        def on_key_event(event):
            if self.running:
                activity = {
                    "type": "keyboard",
                    "timestamp": time.time(),
                    "key": event.name,
                    "event_type": event.event_type
                }
                self.activity_queue.put(activity)
                
        keyboard.hook(on_key_event)
        
    def _monitor_mouse(self):
        """Monitor mouse movements and clicks"""
        def on_mouse_event(event):
            if self.running:
                activity = {
                    "type": "mouse",
                    "timestamp": time.time(),
                    "event_type": event.event_type,
                    "position": event.position
                }
                self.activity_queue.put(activity)
                
        mouse.hook(on_mouse_event)
        
    def _monitor_system(self):
        """Monitor system resources"""
        while self.running:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Record activity
                activity = {
                    "type": "system",
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent
                }
                
                self.activity_queue.put(activity)
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logging.error(f"Error monitoring system: {e}")

class ProgressiveLearner:
    def __init__(self, config: ProgressiveLearningConfig):
        self.config = config
        self.resource_manager = ResourceManager(
            max_memory_gb=config.max_memory_gb,
            max_disk_gb=config.max_disk_gb,
            compression_level=config.compression_level
        )
        
        # Initialize activity monitor
        self.activity_monitor = ActivityMonitor(config)
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-33b-instruct",
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-33b-instruct"
        )
        
        # Create data directory
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize learning thread
        self.learning_thread = None
        self.running = False
        
    def start_learning(self):
        """Start progressive learning"""
        self.running = True
        self.activity_monitor.start_monitoring()
        self.learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True
        )
        self.learning_thread.start()
        
    def stop_learning(self):
        """Stop progressive learning"""
        self.running = False
        self.activity_monitor.stop_monitoring()
        if self.learning_thread:
            self.learning_thread.join()
            
    def _learning_loop(self):
        """Main learning loop"""
        while self.running:
            try:
                # Process activities
                activities = []
                while not self.activity_monitor.activity_queue.empty():
                    activity = self.activity_monitor.activity_queue.get()
                    activities.append(activity)
                    
                if activities:
                    # Convert activities to training data
                    training_data = self._prepare_training_data(activities)
                    
                    # Train on new data
                    self._train_step(training_data)
                    
                    # Save progress
                    self._save_progress()
                    
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                logging.error(f"Error in learning loop: {e}")
                
    def _prepare_training_data(self, activities: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert activities to training data"""
        # Group activities by type
        app_activities = [a for a in activities if a["type"] == "application"]
        screen_activities = [a for a in activities if a["type"] == "screen"]
        keyboard_activities = [a for a in activities if a["type"] == "keyboard"]
        mouse_activities = [a for a in activities if a["type"] == "mouse"]
        system_activities = [a for a in activities if a["type"] == "system"]
        
        # Create training examples
        examples = []
        
        # Application usage patterns
        if app_activities:
            app_text = "User opened applications: " + ", ".join(
                f"{a['app_name']} ({a['window_title']})"
                for a in app_activities
            )
            examples.append(app_text)
            
        # Screen content analysis
        if screen_activities:
            screen_text = "Screen content captured and analyzed"
            examples.append(screen_text)
            
        # User interaction patterns
        if keyboard_activities or mouse_activities:
            interaction_text = "User interactions: " + ", ".join(
                f"{a['event_type']} at {a['position']}"
                for a in mouse_activities
            )
            examples.append(interaction_text)
            
        # System resource usage
        if system_activities:
            system_text = "System resources: " + ", ".join(
                f"CPU: {a['cpu_percent']}%, Memory: {a['memory_percent']}%"
                for a in system_activities
            )
            examples.append(system_text)
            
        # Tokenize examples
        inputs = self.tokenizer(
            examples,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        return inputs
        
    def _train_step(self, training_data: Dict[str, torch.Tensor]):
        """Perform one training step"""
        # Check resources
        if not self.resource_manager.check_resources():
            logging.warning("Skipping training step due to resource constraints")
            return
            
        # Forward pass
        outputs = self.model(
            input_ids=training_data["input_ids"],
            attention_mask=training_data["attention_mask"],
            labels=training_data["input_ids"]
        )
        
        # Backward pass
        loss = outputs.loss
        loss.backward()
        
        # Update weights
        self.model.optimizer.step()
        self.model.optimizer.zero_grad()
        
    def _save_progress(self):
        """Save learning progress"""
        # Save model state
        model_path = Path(self.config.model_dir) / "model_state.pt"
        self.resource_manager.compress_checkpoint(
            self.model.state_dict(),
            str(model_path)
        )
        
        # Save activity data
        data_path = Path(self.config.data_dir) / f"activities_{int(time.time())}.json"
        with open(data_path, "w") as f:
            json.dump(self.last_activity, f)
            
    def get_insights(self) -> Dict[str, Any]:
        """Get insights from learned patterns"""
        insights = {
            "application_usage": self._analyze_app_usage(),
            "interaction_patterns": self._analyze_interactions(),
            "resource_usage": self._analyze_resources()
        }
        return insights
        
    def _analyze_app_usage(self) -> Dict[str, Any]:
        """Analyze application usage patterns"""
        # Implement application usage analysis
        return {}
        
    def _analyze_interactions(self) -> Dict[str, Any]:
        """Analyze user interaction patterns"""
        # Implement interaction pattern analysis
        return {}
        
    def _analyze_resources(self) -> Dict[str, Any]:
        """Analyze system resource usage patterns"""
        # Implement resource usage analysis
        return {} 