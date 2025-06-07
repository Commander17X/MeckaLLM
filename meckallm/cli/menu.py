import os
import sys
import time
import logging
from typing import Dict, List, Optional
import subprocess
from pathlib import Path
import psutil
import GPUtil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich import box
import keyboard
import mouse
from ..learning.progressive_learner import ProgressiveLearner, ProgressiveLearningConfig
from ..learning.autonomous_learner import AutonomousLearner, LearningConfig
from ..optimization.resource_manager import ResourceManager
from .terms import TermsAndCredits

class MeckaLLMMenu:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.running = True
        self.learner = None
        self.resource_manager = ResourceManager()
        self.terms = TermsAndCredits()
        self.terms_accepted = False
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def show_header(self):
        """Display the MeckaLLM header"""
        header = """
        ███╗   ███╗███████╗ ██████╗██╗  ██╗ █████╗ ██╗     ██╗     ██╗███╗   ███╗
        ████╗ ████║██╔════╝██╔════╝██║ ██╔╝██╔══██╗██║     ██║     ██║████╗ ████║
        ██╔████╔██║█████╗  ██║     █████╔╝ ███████║██║     ██║     ██║██╔████╔██║
        ██║╚██╔╝██║██╔══╝  ██║     ██╔═██╗ ██╔══██║██║     ██║     ██║██║╚██╔╝██║
        ██║ ╚═╝ ██║███████╗╚██████╗██║  ██╗██║  ██║███████╗███████╗██║██║ ╚═╝ ██║
        ╚═╝     ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝     ╚═╝
        """
        self.console.print(Panel(header, style="bold blue", box=box.ROUNDED))
        
    def show_system_stats(self):
        """Display system statistics"""
        stats = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        stats.add_column("Metric", style="cyan")
        stats.add_column("Value", style="green")
        
        # CPU Usage
        cpu_percent = psutil.cpu_percent()
        stats.add_row("CPU Usage", f"{cpu_percent}%")
        
        # Memory Usage
        memory = psutil.virtual_memory()
        stats.add_row("Memory Usage", f"{memory.percent}%")
        
        # GPU Usage
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                stats.add_row(f"GPU {gpu.id} Usage", f"{gpu.load*100:.1f}%")
                stats.add_row(f"GPU {gpu.id} Temperature", f"{gpu.temperature}°C")
        except:
            stats.add_row("GPU", "Not available")
            
        self.console.print(Panel(stats, title="System Statistics", border_style="blue"))
        
    def show_menu(self):
        """Display the main menu"""
        menu = Table(show_header=False, box=box.ROUNDED)
        menu.add_column("Option", style="cyan")
        menu.add_column("Description", style="green")
        
        menu.add_row("1", "Start Everything (Progressive + Autonomous Learning)")
        menu.add_row("2", "Start Progressive Learning Only")
        menu.add_row("3", "Start Autonomous Learning Only")
        menu.add_row("4", "Show System Statistics")
        menu.add_row("5", "Show Learning Insights")
        menu.add_row("6", "Configure Settings")
        menu.add_row("7", "Update GitHub Repository")
        menu.add_row("8", "Show Credits")
        menu.add_row("q", "Quit")
        
        self.console.print(Panel(menu, title="Main Menu", border_style="blue"))
        
    def show_progress(self, message: str):
        """Display a progress spinner"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            progress.add_task(description=message, total=None)
            time.sleep(2)
            
    def start_everything(self):
        """Start both learning systems"""
        if not self.terms_accepted:
            self.terms_accepted = self.terms.show_terms()
            if not self.terms_accepted:
                self.console.print("[red]Terms not accepted. Exiting...[/red]")
                return False
                
        self.show_progress("Starting Progressive Learning...")
        prog_config = ProgressiveLearningConfig()
        self.prog_learner = ProgressiveLearner(prog_config)
        self.prog_learner.start_learning()
        
        self.show_progress("Starting Autonomous Learning...")
        auto_config = LearningConfig()
        self.auto_learner = AutonomousLearner(auto_config)
        
        self.console.print("[green]All systems started successfully![/green]")
        return True
        
    def update_github(self):
        """Update the GitHub repository"""
        try:
            self.console.print("[yellow]Updating GitHub repository...[/yellow]")
            
            # Initialize git if not already done
            if not os.path.exists(".git"):
                subprocess.run(["git", "init"], check=True)
                
            # Add all files
            subprocess.run(["git", "add", "."], check=True)
            
            # Commit changes
            subprocess.run(
                ["git", "commit", "-m", "Update MeckaLLM with new features"],
                check=True
            )
            
            # Add remote if not exists
            try:
                subprocess.run(
                    ["git", "remote", "add", "origin", "https://github.com/Commander17X/MeckaLLM.git"],
                    check=True
                )
            except subprocess.CalledProcessError:
                pass  # Remote might already exist
                
            # Set main branch
            subprocess.run(["git", "branch", "-M", "main"], check=True)
            
            # Push changes
            subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
            
            self.console.print("[green]Successfully updated GitHub repository![/green]")
            
        except subprocess.CalledProcessError as e:
            self.console.print(f"[red]Error updating GitHub: {e}[/red]")
            
    def run(self):
        """Run the menu interface"""
        # Show credits first
        self.terms.show_credits()
        
        while self.running:
            self.clear_screen()
            self.show_header()
            self.show_system_stats()
            self.show_menu()
            
            choice = input("\nEnter your choice: ").lower()
            
            if choice == "1":
                if self.start_everything():
                    input("\nPress Enter to continue...")
                
            elif choice == "2":
                if not self.terms_accepted:
                    self.terms_accepted = self.terms.show_terms()
                    if not self.terms_accepted:
                        continue
                self.show_progress("Starting Progressive Learning...")
                config = ProgressiveLearningConfig()
                self.learner = ProgressiveLearner(config)
                self.learner.start_learning()
                self.console.print("[green]Progressive Learning started![/green]")
                input("\nPress Enter to continue...")
                
            elif choice == "3":
                if not self.terms_accepted:
                    self.terms_accepted = self.terms.show_terms()
                    if not self.terms_accepted:
                        continue
                self.show_progress("Starting Autonomous Learning...")
                config = LearningConfig()
                self.learner = AutonomousLearner(config)
                self.console.print("[green]Autonomous Learning started![/green]")
                input("\nPress Enter to continue...")
                
            elif choice == "4":
                self.show_system_stats()
                input("\nPress Enter to continue...")
                
            elif choice == "5":
                if hasattr(self, 'prog_learner') or hasattr(self, 'auto_learner'):
                    insights = {}
                    if hasattr(self, 'prog_learner'):
                        insights['progressive'] = self.prog_learner.get_insights()
                    if hasattr(self, 'auto_learner'):
                        insights['autonomous'] = self.auto_learner.get_insights()
                    self.console.print(Panel(str(insights), title="Learning Insights"))
                else:
                    self.console.print("[yellow]No active learning session[/yellow]")
                input("\nPress Enter to continue...")
                
            elif choice == "6":
                self.show_progress("Loading settings...")
                # TODO: Implement settings configuration
                self.console.print("[yellow]Settings configuration coming soon![/yellow]")
                input("\nPress Enter to continue...")
                
            elif choice == "7":
                self.update_github()
                input("\nPress Enter to continue...")
                
            elif choice == "8":
                self.terms.show_credits()
                
            elif choice == "q":
                if hasattr(self, 'prog_learner'):
                    self.prog_learner.stop_learning()
                if hasattr(self, 'auto_learner'):
                    self.auto_learner.cleanup()
                self.running = False
                self.console.print("[yellow]Goodbye![/yellow]")
                
            else:
                self.console.print("[red]Invalid choice![/red]")
                time.sleep(1)
                
if __name__ == "__main__":
    menu = MeckaLLMMenu()
    menu.run() 