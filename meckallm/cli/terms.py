from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

class TermsAndCredits:
    def __init__(self):
        self.console = Console()
        
    def show_terms(self) -> bool:
        """Display terms of service and get user agreement"""
        terms = """
        MeckaLLM Terms of Service

        1. Usage and Privacy:
           - MeckaLLM monitors system activities for learning purposes
           - All data is stored locally on your machine
           - No data is sent to external servers without consent

        2. Resource Usage:
           - MeckaLLM optimizes resource usage
           - System monitoring may impact performance
           - You can stop monitoring at any time

        3. Learning and Adaptation:
           - MeckaLLM learns from your computer usage
           - Learning is continuous and automatic
           - You can view and control learning parameters

        4. GitHub Integration:
           - Code is stored in your GitHub repository
           - You maintain full control of your data
           - Updates are pushed only with your permission

        By accepting these terms, you agree to allow MeckaLLM to:
        - Monitor system activities
        - Learn from your usage patterns
        - Store data locally
        - Update GitHub repository when requested
        """
        
        self.console.print(Panel(terms, title="Terms of Service", border_style="blue", box=box.ROUNDED))
        
        while True:
            response = input("\nDo you agree to these terms? (yes/no): ").lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                self.console.print("[red]Please answer 'yes' or 'no'[/red]")
                
    def show_credits(self):
        """Display credits and acknowledgments"""
        credits = """
        MeckaLLM - Advanced AI Learning System
        ======================================

        Created by: Commander17X
        GitHub: https://github.com/Commander17X

        Special Thanks:
        - DeepSeek AI for the base model
        - The open-source community
        - All contributors and testers

        Technologies Used:
        - PyTorch
        - Transformers
        - Rich CLI
        - Various open-source libraries

        License: MIT
        """
        
        self.console.print(Panel(credits, title="Credits", border_style="green", box=box.ROUNDED))
        input("\nPress Enter to continue...") 