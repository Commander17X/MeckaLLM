import os
import sys
import time
import logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..monitoring.autonomous_monitor import AutonomousMonitor, AutonomousConfig

def main():
    """Run the autonomous monitoring demo."""
    console = Console()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    console.print(Panel.fit(
        "[bold blue]MeckaLLM Autonomous Monitoring Demo[/bold blue]\n"
        "This demo will:\n"
        "• Monitor your system activities\n"
        "• Learn from safe content\n"
        "• Maintain content safety\n"
        "• Record activity history",
        title="Welcome"
    ))
    
    # Initialize monitor
    config = AutonomousConfig(
        check_interval=2,  # Check every 2 seconds
        max_history=100    # Keep last 100 activities
    )
    monitor = AutonomousMonitor(config)
    
    try:
        # Start monitoring
        monitor.start()
        
        # Run for 5 minutes
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Monitoring...", total=300)  # 5 minutes
            
            while not progress.finished:
                time.sleep(1)
                progress.update(task, advance=1)
                
                # Display stats every 30 seconds
                if progress.tasks[0].completed % 30 == 0:
                    monitor.display_stats()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping monitoring...[/yellow]")
    finally:
        # Stop monitoring
        monitor.stop()
        
        # Display final stats
        console.print("\n[bold]Final Statistics:[/bold]")
        monitor.display_stats()

if __name__ == "__main__":
    main() 