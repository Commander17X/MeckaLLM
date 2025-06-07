import os
import sys
import logging
from pathlib import Path
import time

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from meckallm.monitoring.learning_monitor import LearningMonitor, MonitoringConfig
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the monitoring system demo."""
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
            log_file="monitoring.log",
            save_interval=5,
            max_violations=3,
            alert_threshold=2
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
                "name": "Normal Content",
                "content": "This is a normal text about programming and technology.",
                "source": "github.com/example/repo"
            },
            {
                "name": "Sensitive Information",
                "content": "API_KEY=1234567890abcdef\nPASSWORD=secret123",
                "source": "config.txt"
            },
            {
                "name": "Adult Content",
                "content": "This is an adult content website with explicit material.",
                "source": "adult-content.com"
            },
            {
                "name": "Political Content",
                "content": "Political campaign strategies and election manipulation techniques.",
                "source": "political-extremist.org"
            },
            {
                "name": "Financial Content",
                "content": "Stock market manipulation and cryptocurrency trading secrets.",
                "source": "financial-scam.com"
            },
            {
                "name": "Hate Speech",
                "content": "Content containing discriminatory and hateful language.",
                "source": "hate-speech.net"
            },
            {
                "name": "Violence",
                "content": "Instructions for violent activities and weapon usage.",
                "source": "violence-guide.com"
            },
            {
                "name": "Illegal Activities",
                "content": "How to perform illegal activities and bypass security.",
                "source": "illegal-market.com"
            },
            {
                "name": "Obfuscated URL",
                "content": "Visit hxxp://unbl0ck-s1te.com to bypass restrictions",
                "source": "bypass-attempt.txt"
            },
            {
                "name": "Shortened URL",
                "content": "Check out this link: bit.ly/suspicious-content",
                "source": "short-url.txt"
            },
            {
                "name": "IP Address URL",
                "content": "Access content at http://192.168.1.1/restricted",
                "source": "ip-url.txt"
            },
            {
                "name": "Suspicious TLD",
                "content": "Visit website.example.xyz for more information",
                "source": "suspicious-tld.txt"
            },
            {
                "name": "Content Obfuscation",
                "content": "Use pr0xy-s3rv1c3 to access restricted content",
                "source": "obfuscated.txt"
            },
            {
                "name": "Multiple Bypass Attempts",
                "content": "Try vpn-service.com or tor-network.org to access blocked sites",
                "source": "multiple-bypass.txt"
            }
        ]
        
        # Create progress display
        with Progress() as progress:
            task = progress.add_task("[cyan]Testing monitoring system...", total=len(test_cases))
            
            # Test each case
            for case in test_cases:
                result = monitor.monitor_content(case["content"], case["source"])
                
                # Display result
                status = "[green]✓ Allowed" if result["allowed"] else "[red]✗ Blocked"
                console.print(f"\n{status} - {case['name']}")
                
                if not result["allowed"]:
                    violations = result["violations"]
                    for category, details in violations.items():
                        console.print(f"  • {category}:")
                        for detail in details:
                            console.print(f"    - {detail}")
                
                progress.update(task, advance=1)
                time.sleep(0.5)  # Add small delay for better visualization
        
        # Display final statistics
        stats = monitor.get_stats()
        console.print("\n[bold cyan]Monitoring Statistics:[/bold cyan]")
        console.print(f"Total checks: {stats['total_checks']}")
        console.print(f"Violations: {stats['violations']}")
        console.print(f"Blocked content: {stats['blocked_content']}")
        
        # Display blacklist summary
        blacklist_summary = monitor.get_blacklist_summary()
        console.print("\n[bold cyan]Blacklist Summary:[/bold cyan]")
        for category, count in blacklist_summary.items():
            console.print(f"{category}: {count} items")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
    finally:
        console.print("\n[bold green]Demo completed![/bold green]")

if __name__ == "__main__":
    main() 