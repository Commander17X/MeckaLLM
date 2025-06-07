import os
import sys
import time
import json
import logging
import psutil
import threading
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

from .blacklist import BlacklistMonitor, BlacklistConfig
from .learning_monitor import LearningMonitor, MonitoringConfig

@dataclass
class AutonomousConfig:
    """Configuration for autonomous monitoring."""
    log_file: str = "autonomous_monitor.log"
    activity_log: str = "activity_history.json"
    learning_log: str = "learning_history.json"
    check_interval: int = 5  # seconds
    max_history: int = 1000
    allowed_processes: Set[str] = field(default_factory=lambda: {
        "chrome.exe", "firefox.exe", "msedge.exe", "python.exe",
        "code.exe", "notepad.exe", "explorer.exe", "powershell.exe"
    })
    learning_threshold: float = 0.8  # confidence threshold for learning

class AutonomousMonitor:
    """Autonomous monitoring system that observes user activities and learns from them."""
    
    def __init__(self, config: Optional[AutonomousConfig] = None):
        self.config = config or AutonomousConfig()
        self.console = Console()
        self._setup_logging()
        
        # Initialize monitors
        self.blacklist_monitor = BlacklistMonitor()
        self.learning_monitor = LearningMonitor(MonitoringConfig())
        
        # Initialize state
        self.running = False
        self.activity_history: List[Dict] = []
        self.learning_history: List[Dict] = []
        self.observed_patterns: Dict[str, int] = {}
        self.safe_patterns: Set[str] = set()
        
        # Load history
        self._load_history()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            filename=self.config.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_history(self):
        """Load activity and learning history."""
        try:
            if os.path.exists(self.config.activity_log):
                with open(self.config.activity_log, 'r') as f:
                    self.activity_history = json.load(f)
            if os.path.exists(self.config.learning_log):
                with open(self.config.learning_log, 'r') as f:
                    self.learning_history = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading history: {str(e)}")
    
    def _save_history(self):
        """Save activity and learning history."""
        try:
            with open(self.config.activity_log, 'w') as f:
                json.dump(self.activity_history[-self.config.max_history:], f, indent=2)
            with open(self.config.learning_log, 'w') as f:
                json.dump(self.learning_history[-self.config.max_history:], f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving history: {str(e)}")
    
    def _get_active_processes(self) -> List[Dict]:
        """Get information about active processes."""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                if proc.info['name'] in self.config.allowed_processes:
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': proc.info['cmdline'],
                        'create_time': proc.info['create_time']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return processes
    
    def _extract_patterns(self, content: str) -> Dict[str, int]:
        """Extract patterns from content for learning."""
        patterns = {}
        
        # Extract URLs
        urls = re.findall(r'https?://\S+|www\.\S+', content)
        for url in urls:
            patterns[f'url:{url}'] = patterns.get(f'url:{url}', 0) + 1
        
        # Extract file paths
        paths = re.findall(r'[A-Za-z]:\\[\\\S|*\S]?.*|/[\\\S|*\S]?.*', content)
        for path in paths:
            patterns[f'path:{path}'] = patterns.get(f'path:{path}', 0) + 1
        
        # Extract command patterns
        commands = re.findall(r'[a-zA-Z0-9_]+\([^)]*\)', content)
        for cmd in commands:
            patterns[f'cmd:{cmd}'] = patterns.get(f'cmd:{cmd}', 0) + 1
        
        return patterns
    
    def _analyze_content(self, content: str, source: str) -> Dict:
        """Analyze content for learning and safety."""
        # Check against blacklist
        violations = self.blacklist_monitor.get_violations(content, source)
        
        # Analyze patterns
        patterns = self._extract_patterns(content)
        
        return {
            'violations': violations,
            'patterns': patterns,
            'timestamp': time.time(),
            'source': source
        }
    
    def _update_learning(self, analysis: Dict):
        """Update learning based on content analysis."""
        if not any(analysis['violations'].values()):
            # Content is safe, update learning
            self.learning_history.append({
                'timestamp': time.time(),
                'patterns': analysis['patterns'],
                'source': analysis['source']
            })
            
            # Update safe patterns
            for pattern, count in analysis['patterns'].items():
                if count >= 2:  # Pattern appears multiple times
                    self.safe_patterns.add(pattern)
            
            self._save_history()
    
    def _monitor_activity(self):
        """Monitor user activity and learn from it."""
        while self.running:
            try:
                # Get active processes
                processes = self._get_active_processes()
                
                # Record activity
                activity = {
                    'timestamp': time.time(),
                    'processes': processes
                }
                self.activity_history.append(activity)
                
                # Analyze and learn
                for proc in processes:
                    if proc['cmdline']:
                        content = ' '.join(proc['cmdline'])
                        analysis = self._analyze_content(content, proc['name'])
                        self._update_learning(analysis)
                
                # Save history periodically
                if len(self.activity_history) % 10 == 0:
                    self._save_history()
                
                time.sleep(self.config.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring: {str(e)}")
                time.sleep(self.config.check_interval)
    
    def start(self):
        """Start autonomous monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_activity)
        self.monitor_thread.start()
        
        self.console.print(Panel.fit(
            "[bold green]Autonomous Monitoring Started[/bold green]\n"
            "• Monitoring user activities\n"
            "• Learning from safe content\n"
            "• Maintaining content safety\n"
            "• Recording activity history",
            title="MeckaLLM Autonomous Monitor"
        ))
    
    def stop(self):
        """Stop autonomous monitoring."""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        self._save_history()
        
        self.console.print(Panel.fit(
            "[bold red]Autonomous Monitoring Stopped[/bold red]\n"
            f"• Activities recorded: {len(self.activity_history)}\n"
            f"• Learning events: {len(self.learning_history)}\n"
            f"• Safe patterns learned: {len(self.safe_patterns)}",
            title="MeckaLLM Autonomous Monitor"
        ))
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        return {
            'activities_recorded': len(self.activity_history),
            'learning_events': len(self.learning_history),
            'safe_patterns': len(self.safe_patterns),
            'observed_patterns': len(self.observed_patterns),
            'last_activity': self.activity_history[-1] if self.activity_history else None,
            'last_learning': self.learning_history[-1] if self.learning_history else None
        }
    
    def display_stats(self):
        """Display monitoring statistics in a rich table."""
        stats = self.get_stats()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Activities Recorded", str(stats['activities_recorded']))
        table.add_row("Learning Events", str(stats['learning_events']))
        table.add_row("Safe Patterns", str(stats['safe_patterns']))
        table.add_row("Observed Patterns", str(stats['observed_patterns']))
        
        if stats['last_activity']:
            table.add_row("Last Activity", datetime.fromtimestamp(
                stats['last_activity']['timestamp']
            ).strftime('%Y-%m-%d %H:%M:%S'))
        
        if stats['last_learning']:
            table.add_row("Last Learning", datetime.fromtimestamp(
                stats['last_learning']['timestamp']
            ).strftime('%Y-%m-%d %H:%M:%S'))
        
        self.console.print(Panel(table, title="Monitoring Statistics")) 