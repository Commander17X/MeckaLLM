import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from meckallm.models.deepseek_integration import DeepSeekModel, DeepSeekConfig
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    console = Console()
    
    # Display welcome message
    console.print(Panel.fit(
        "[bold blue]DeepSeek Model Integration Demo[/bold blue]\n"
        "This demo showcases the capabilities of the DeepSeek model integration.",
        title="Welcome"
    ))
    
    try:
        # Initialize model with custom config
        config = DeepSeekConfig(
            model_name="deepseek-ai/deepseek-coder-33b-instruct",
            temperature=0.7,
            max_length=2048
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(description="Loading DeepSeek model...", total=None)
            model = DeepSeekModel(config)
        
        # Demo 1: Code Generation
        console.print("\n[bold green]Demo 1: Code Generation[/bold green]")
        prompt = "Create a function to calculate the Fibonacci sequence using dynamic programming"
        console.print(f"\nPrompt: {prompt}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(description="Generating code...", total=None)
            code = model.generate_code(prompt)
        
        console.print(Panel(code, title="Generated Code"))
        
        # Demo 2: Code Analysis
        console.print("\n[bold green]Demo 2: Code Analysis[/bold green]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(description="Analyzing code...", total=None)
            analysis = model.analyze_code(code)
        
        console.print(Panel(
            f"Quality Score: {analysis['quality_score']}\n\n"
            f"Security Issues:\n" + "\n".join(f"- {issue}" for issue in analysis['security_issues']) + "\n\n"
            f"Optimizations:\n" + "\n".join(f"- {opt}" for opt in analysis['optimizations']) + "\n\n"
            f"Best Practices:\n" + "\n".join(f"- {bp}" for bp in analysis['best_practices']) + "\n\n"
            f"Potential Bugs:\n" + "\n".join(f"- {bug}" for bug in analysis['potential_bugs']),
            title="Code Analysis"
        ))
        
        # Demo 3: Text Generation
        console.print("\n[bold green]Demo 3: Text Generation[/bold green]")
        prompt = "Explain the concept of quantum computing in simple terms"
        console.print(f"\nPrompt: {prompt}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task(description="Generating text...", total=None)
            text = model.generate(prompt)
        
        console.print(Panel(text, title="Generated Text"))
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
    finally:
        # Cleanup
        if 'model' in locals():
            model.cleanup()
        
        console.print("\n[bold green]Demo completed![/bold green]")

if __name__ == "__main__":
    main() 