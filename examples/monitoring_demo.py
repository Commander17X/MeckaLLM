import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from meckallm.monitoring.learning_monitor import LearningMonitor, MonitoringConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    console = Console()
    
    # Display welcome message
    console.print(Panel.fit(
        "[bold blue]Learning Monitor Demo[/bold blue]\n"
        "This demo showcases the content monitoring and blacklist system.",
        title="Welcome"
    ))
    
    try:
        # Initialize monitor
        config = MonitoringConfig(
            log_file="demo_monitor.log",
            save_interval=60,  # 1 minute for demo
            max_violations=5,
            alert_threshold=3
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(description="Initializing monitor...", total=None)
            monitor = LearningMonitor(config)
        
        # Test cases
        test_cases = [
            {
                "content": "This is a normal text about programming",
                "source": "test_case_1",
                "description": "Normal content"
            },
            {
                "content": "Here's a password: secret123",
                "source": "test_case_2",
                "description": "Password in content"
            },
            {
                "content": "Download cracked software from malicious.com",
                "source": "test_case_3",
                "description": "Malicious domain"
            },
            {
                "content": "Run this .exe file to hack the system",
                "source": "test_case_4",
                "description": "Executable file"
            }
        ]
        
        # Run test cases
        for test in test_cases:
            console.print(f"\n[bold green]Testing: {test['description']}[/bold green]")
            console.print(f"Content: {test['content']}")
            
            result = monitor.monitor_content(test['content'], test['source'])
            
            # Display results
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Allowed", str(result['allowed']))
            table.add_row("Violations Count", str(result['violations_count']))
            
            if result['violations']:
                for category, items in result['violations'].items():
                    if items:
                        table.add_row(f"{category} violations", ", ".join(items))
            
            console.print(table)
        
        # Show final stats
        console.print("\n[bold blue]Final Statistics[/bold blue]")
        stats = monitor.get_stats()
        
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        for key, value in stats['monitoring_stats'].items():
            if key != 'violation_history':
                stats_table.add_row(key, str(value))
        
        console.print(stats_table)
        
        # Show blacklist summary
        console.print("\n[bold blue]Blacklist Summary[/bold blue]")
        blacklist_table = Table(show_header=True, header_style="bold magenta")
        blacklist_table.add_column("Category", style="cyan")
        blacklist_table.add_column("Count", style="green")
        
        for category, count in stats['blacklist_summary'].items():
            blacklist_table.add_row(category, str(count))
        
        console.print(blacklist_table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
    finally:
        console.print("\n[bold green]Demo completed![/bold green]")

if __name__ == "__main__":
    main() 