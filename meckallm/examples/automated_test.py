import os
import sys
import time
import random
import json
import pyautogui
import keyboard
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..monitoring.autonomous_monitor import AutonomousMonitor, AutonomousConfig

class AutomatedTester:
    """Automated testing system for the autonomous monitor with progressive learning."""
    
    def __init__(self):
        self.console = Console()
        self.monitor = AutonomousMonitor(AutonomousConfig(
            check_interval=1,  # Check every second
            max_history=1000
        ))
        
        # Learning categories and their associated content
        self.learning_categories = {
            'programming': {
                'websites': [
                    "https://www.github.com",
                    "https://www.stackoverflow.com",
                    "https://www.python.org",
                    "https://www.w3schools.com",
                    "https://www.codecademy.com"
                ],
                'files': [
                    "main.py",
                    "utils.py",
                    "config.py",
                    "requirements.txt",
                    "README.md"
                ],
                'commands': [
                    "python main.py",
                    "pip install -r requirements.txt",
                    "git commit -m 'update'",
                    "python -m pytest",
                    "black ."
                ]
            },
            'research': {
                'websites': [
                    "https://www.wikipedia.org",
                    "https://scholar.google.com",
                    "https://www.researchgate.net",
                    "https://www.sciencedirect.com",
                    "https://www.arxiv.org"
                ],
                'files': [
                    "research_paper.md",
                    "data_analysis.py",
                    "results.csv",
                    "literature_review.txt",
                    "methodology.pdf"
                ],
                'commands': [
                    "python data_analysis.py",
                    "jupyter notebook",
                    "python -m pip install pandas numpy",
                    "git pull origin main",
                    "python -m pytest tests/"
                ]
            },
            'development': {
                'websites': [
                    "https://www.docker.com",
                    "https://www.kubernetes.io",
                    "https://www.aws.amazon.com",
                    "https://www.azure.microsoft.com",
                    "https://www.digitalocean.com"
                ],
                'files': [
                    "Dockerfile",
                    "docker-compose.yml",
                    "kubernetes.yaml",
                    "terraform.tf",
                    "cloud_config.json"
                ],
                'commands': [
                    "docker build -t app .",
                    "kubectl apply -f kubernetes.yaml",
                    "terraform init",
                    "aws s3 ls",
                    "docker-compose up"
                ]
            }
        }
        
        # Progressive learning stages
        self.learning_stages = [
            'basic',      # Simple activities
            'intermediate',  # More complex patterns
            'advanced',   # Complex combinations
            'expert'      # Edge cases and optimizations
        ]
        
        # Blacklisted content (for testing violations)
        self.blacklisted_content = [
            "password123",
            "api_key_123456",
            "credit_card_1234",
            "sensitive_data",
            "secret_token",
            "private_key",
            "aws_access_key",
            "database_password"
        ]
        
        # Learning progress tracking
        self.learning_progress = {
            'stage': 0,
            'patterns_learned': set(),
            'categories_covered': set(),
            'violations_detected': 0,
            'safe_patterns': set()
        }
    
    def _type_slowly(self, text: str, interval: float = 0.1):
        """Type text slowly to simulate human typing."""
        for char in text:
            keyboard.write(char)
            time.sleep(interval)
    
    def _get_current_stage_content(self):
        """Get content appropriate for the current learning stage."""
        stage = self.learning_stages[self.learning_progress['stage']]
        category = random.choice(list(self.learning_categories.keys()))
        
        if stage == 'basic':
            return {
                'websites': [self.learning_categories[category]['websites'][0]],
                'files': [self.learning_categories[category]['files'][0]],
                'commands': [self.learning_categories[category]['commands'][0]]
            }
        elif stage == 'intermediate':
            return {
                'websites': self.learning_categories[category]['websites'][:2],
                'files': self.learning_categories[category]['files'][:2],
                'commands': self.learning_categories[category]['commands'][:2]
            }
        elif stage == 'advanced':
            return {
                'websites': self.learning_categories[category]['websites'][:3],
                'files': self.learning_categories[category]['files'][:3],
                'commands': self.learning_categories[category]['commands'][:3]
            }
        else:  # expert
            return {
                'websites': self.learning_categories[category]['websites'],
                'files': self.learning_categories[category]['files'],
                'commands': self.learning_categories[category]['commands']
            }
    
    def _simulate_browser_activity(self):
        """Simulate web browser activity with progressive learning."""
        self.console.print(f"[yellow]Simulating browser activity (Stage: {self.learning_stages[self.learning_progress['stage']]})...[/yellow]")
        
        # Open browser
        pyautogui.hotkey('win', 'r')
        time.sleep(1)
        self._type_slowly("chrome")
        keyboard.press('enter')
        time.sleep(2)
        
        # Get content for current stage
        content = self._get_current_stage_content()
        
        # Visit websites
        for site in content['websites']:
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.5)
            self._type_slowly(site)
            keyboard.press('enter')
            time.sleep(random.uniform(2, 4))
            
            # Update learning progress
            self.learning_progress['patterns_learned'].add(f'website:{site}')
        
        # Occasionally try blacklisted content
        if random.random() < 0.2:  # 20% chance
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.5)
            self._type_slowly("https://suspicious-site.xyz")
            keyboard.press('enter')
            time.sleep(2)
            self.learning_progress['violations_detected'] += 1
    
    def _simulate_file_operations(self):
        """Simulate file operations with progressive learning."""
        self.console.print(f"[yellow]Simulating file operations (Stage: {self.learning_stages[self.learning_progress['stage']]})...[/yellow]")
        
        # Open Notepad
        pyautogui.hotkey('win', 'r')
        time.sleep(1)
        self._type_slowly("notepad")
        keyboard.press('enter')
        time.sleep(2)
        
        # Get content for current stage
        content = self._get_current_stage_content()
        
        # Create and edit files
        for file in content['files']:
            self._type_slowly(f"Creating {file}...\n")
            time.sleep(1)
            
            # Add some content based on file type
            if file.endswith('.py'):
                self._type_slowly("def main():\n    print('Hello, World!')\n")
            elif file.endswith('.md'):
                self._type_slowly("# Documentation\n\nThis is a test file.\n")
            elif file.endswith('.json'):
                self._type_slowly('{"key": "value"}\n')
            
            # Update learning progress
            self.learning_progress['patterns_learned'].add(f'file:{file}')
        
        # Occasionally try blacklisted content
        if random.random() < 0.2:  # 20% chance
            self._type_slowly("\n".join(random.sample(self.blacklisted_content, 2)))
            time.sleep(2)
            self.learning_progress['violations_detected'] += 1
        
        # Save and close
        pyautogui.hotkey('ctrl', 's')
        time.sleep(1)
        pyautogui.hotkey('alt', 'f4')
    
    def _simulate_terminal_activity(self):
        """Simulate terminal/command line activity with progressive learning."""
        self.console.print(f"[yellow]Simulating terminal activity (Stage: {self.learning_stages[self.learning_progress['stage']]})...[/yellow]")
        
        # Open PowerShell
        pyautogui.hotkey('win', 'r')
        time.sleep(1)
        self._type_slowly("powershell")
        keyboard.press('enter')
        time.sleep(2)
        
        # Get content for current stage
        content = self._get_current_stage_content()
        
        # Run commands
        for cmd in content['commands']:
            self._type_slowly(cmd)
            keyboard.press('enter')
            time.sleep(random.uniform(1, 2))
            
            # Update learning progress
            self.learning_progress['patterns_learned'].add(f'command:{cmd}')
        
        # Occasionally try blacklisted command
        if random.random() < 0.2:  # 20% chance
            self._type_slowly("echo " + random.choice(self.blacklisted_content))
            keyboard.press('enter')
            time.sleep(2)
            self.learning_progress['violations_detected'] += 1
        
        # Close terminal
        self._type_slowly("exit")
        keyboard.press('enter')
    
    def _update_learning_stage(self):
        """Update the learning stage based on progress."""
        current_stage = self.learning_progress['stage']
        patterns_learned = len(self.learning_progress['patterns_learned'])
        
        # Progress to next stage if enough patterns are learned
        if patterns_learned >= (current_stage + 1) * 10:
            if current_stage < len(self.learning_stages) - 1:
                self.learning_progress['stage'] += 1
                self.console.print(f"[green]Progressing to {self.learning_stages[self.learning_progress['stage']]} stage![/green]")
    
    def _display_learning_progress(self):
        """Display current learning progress."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Current Stage", self.learning_stages[self.learning_progress['stage']])
        table.add_row("Patterns Learned", str(len(self.learning_progress['patterns_learned'])))
        table.add_row("Categories Covered", str(len(self.learning_progress['categories_covered'])))
        table.add_row("Violations Detected", str(self.learning_progress['violations_detected']))
        table.add_row("Safe Patterns", str(len(self.learning_progress['safe_patterns'])))
        
        self.console.print(Panel(table, title="Learning Progress"))
    
    def run_tests(self, duration: int = 300):
        """Run automated tests with progressive learning."""
        self.console.print(Panel.fit(
            "[bold blue]Automated Testing with Progressive Learning Started[/bold blue]\n"
            "• Simulating browser activity\n"
            "• Simulating file operations\n"
            "• Simulating terminal activity\n"
            "• Progressive learning and monitoring",
            title="MeckaLLM Automated Testing"
        ))
        
        # Start monitoring
        self.monitor.start()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Running tests...", total=duration)
                
                start_time = time.time()
                while time.time() - start_time < duration:
                    # Simulate different activities
                    self._simulate_browser_activity()
                    self._simulate_file_operations()
                    self._simulate_terminal_activity()
                    
                    # Update learning stage
                    self._update_learning_stage()
                    
                    # Display progress
                    self._display_learning_progress()
                    self.monitor.display_stats()
                    
                    # Update progress
                    elapsed = time.time() - start_time
                    progress.update(task, completed=min(elapsed, duration))
                    
                    # Small delay between cycles
                    time.sleep(5)
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Stopping tests...[/yellow]")
        finally:
            # Stop monitoring
            self.monitor.stop()
            
            # Save learning progress
            with open('learning_progress.json', 'w') as f:
                json.dump({
                    'stage': self.learning_stages[self.learning_progress['stage']],
                    'patterns_learned': list(self.learning_progress['patterns_learned']),
                    'categories_covered': list(self.learning_progress['categories_covered']),
                    'violations_detected': self.learning_progress['violations_detected'],
                    'safe_patterns': list(self.learning_progress['safe_patterns']),
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            
            # Display final stats
            self.console.print("\n[bold]Final Statistics:[/bold]")
            self._display_learning_progress()
            self.monitor.display_stats()

def main():
    """Run the automated testing."""
    tester = AutomatedTester()
    tester.run_tests(duration=300)  # Run for 5 minutes

if __name__ == "__main__":
    main() 