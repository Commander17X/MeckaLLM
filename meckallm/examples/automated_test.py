import os
import sys
import time
import random
import json
import pyautogui
import keyboard
import speech_recognition as sr
import webbrowser
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..monitoring.autonomous_monitor import AutonomousMonitor, AutonomousConfig
from .voice_control import VoiceControl
from .app_control import AppControl
from .email_control import EmailControl
from .discord_control import DiscordControl
from .learning_manager import LearningManager
from .web_crawler import WebCrawler
from .content_analyzer import ContentAnalyzer
from .video_analyzer import VideoAnalyzer
from .brave_youtube import BraveYouTube

class AutomatedTester:
    """Automated testing system for the autonomous monitor with progressive learning."""
    
    def __init__(self):
        self.console = Console()
        self.monitor = AutonomousMonitor(AutonomousConfig(
            check_interval=1,  # Check every second
            max_history=1000
        ))
        
        # Initialize components
        self.voice_control = VoiceControl(self.console)
        self.app_control = AppControl(self.console)
        self.email_control = EmailControl(self.console)
        self.discord_control = DiscordControl(self.console)
        self.learning_manager = LearningManager(self.console)
        self.web_crawler = WebCrawler(self.console)
        self.content_analyzer = ContentAnalyzer(self.console)
        self.video_analyzer = VideoAnalyzer(self.console)
        self.brave_youtube = BraveYouTube(self.console)
        
        # Load learning data
        self.learning_manager.load_learning_data()
        
    def _handle_voice_prompt(self, prompt: str, language: str):
        """Handle voice prompts with user intervention."""
        try:
            # Store prompt in history
            self.learning_manager.add_prompt_to_history(prompt, language)
            
            # Check for app control commands
            if self.app_control.is_app_control_prompt(prompt):
                return self.app_control.handle_app_control_prompt(prompt, language)
            
            # Check for email control commands
            if self.email_control.is_email_control_prompt(prompt):
                return self.email_control.handle_email_control_prompt(prompt, language)
            
            # Check for Discord control commands
            if self.discord_control.is_discord_control_prompt(prompt):
                return self.discord_control.handle_discord_control_prompt(prompt, language)
            
            # Check for video analysis commands
            if self.video_analyzer.is_video_control_prompt(prompt):
                return self.video_analyzer.handle_video_control_prompt(prompt, language)
            
            # Check for Brave/YouTube commands
            if self.brave_youtube.is_brave_youtube_prompt(prompt):
                return self.brave_youtube.handle_brave_youtube_prompt(prompt, language)
            
            # Handle other prompts
            return self.learning_manager.handle_general_prompt(prompt, language)
            
        except Exception as e:
            self.learning_manager.add_error(f"Failed to handle voice prompt: {str(e)}")
            return False, "prompt handling failed"

    def _listen_for_command(self):
        """Listen for voice commands with prompt handling."""
        return self.voice_control.listen_for_command(self._handle_voice_prompt)

    def run_tests(self, duration: int = 300):
        """Run automated tests with progressive learning."""
        self.console.print(Panel.fit(
            "[bold blue]Automated Testing with Progressive Learning Started[/bold blue]\n"
            "• Simulating browser activity\n"
            "• Simulating file operations\n"
            "• Simulating terminal activity\n"
            "• Progressive learning and monitoring\n"
            "• Voice command processing\n"
            "• Multi-language support\n"
            "• Self-learning capabilities\n"
            "• Discord integration\n"
            "• Video analysis and truth verification\n"
            "• Brave browser and YouTube integration",
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
                    # Listen for voice commands
                    command, language = self._listen_for_command()
                    if command and language:
                        self.learning_manager.process_voice_command(command, language)
                    
                    # Simulate different activities
                    self.learning_manager.simulate_activities()
                    
                    # Update learning stage
                    self.learning_manager.update_learning_stage()
                    
                    # Display progress
                    self.learning_manager.display_learning_progress()
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
            
            # Close Brave browser
            self.brave_youtube.close()
            
            # Save all learning data
            self.learning_manager.save_learning_data()
            
            # Display final stats
            self.console.print("\n[bold]Final Statistics:[/bold]")
            self.learning_manager.display_learning_progress()
            self.monitor.display_stats()

def main():
    """Run the automated testing."""
    tester = AutomatedTester()
    tester.run_tests(duration=300)  # Run for 5 minutes

if __name__ == "__main__":
    main() 