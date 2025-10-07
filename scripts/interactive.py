"""Interactive CLI for querying insurance policies with Ollama"""

import sys
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
import ollama

from insurance_rag import InsurancePolicyRAG
from insurance_rag.config import get_settings


def check_ollama():
    """Check if Ollama is running"""
    try:
        ollama.list()
        return True
    except:
        return False


def main():
    """Run interactive CLI"""
    load_dotenv()
    console = Console()

    console.print(
        Panel.fit(
            "🏥 [bold cyan]Insurance Policy RAG System[/bold cyan]\n"
            "🤖 Powered by Local Ollama (Free & Private)\n"
            "Interactive Query Interface",
            border_style="cyan",
        )
    )

    # Check Ollama
    if not check_ollama():
        console.print("\n[red]❌ Ollama is not running![/red]")
        console.print("\nPlease start Ollama:")
        console.print("  1. Install: https://ollama.com/download")
        console.print("  2. Start: [cyan]ollama serve[/cyan]")
        console.print("  3. Pull model: [cyan]ollama pull gemma2:2b[/cyan]")
        return 1

    # Initialize
    try:
        settings = get_settings()
        console.print(f"\n🔧 Using model: [cyan]{settings.ollama_model}[/cyan]")
        rag = InsurancePolicyRAG()
    except Exception as e:
        console.print(f"[red]❌ Initialization error: {e}[/red]")
        console.print("\nMake sure the model is available:")
        console.print(f"  [cyan]ollama pull {settings.ollama_model}[/cyan]")
        return 1

    # Load policy
    pdf_path = settings.data_dir / "insurance_policy.pdf"

    if not pdf_path.exists():
        console.print(f"\n[red]❌ Policy not found: {pdf_path}[/red]")
        return 1

    try:
        rag.load_policy(pdf_path)
    except Exception as e:
        console.print(f"[red]❌ Error loading policy: {e}[/red]")
        return 1

    # Interactive loop
    console.print("\n[bold green]Ready![/bold green] Ask questions about your policy")
    console.print("[dim](type 'quit', 'exit', or 'q' to exit)[/dim]\n")

    while True:
        try:
            query = Prompt.ask("[bold yellow]Your question[/bold yellow]").strip()

            if query.lower() in ["quit", "exit", "q"]:
                console.print("\n[cyan]👋 Goodbye![/cyan]")
                break

            if not query:
                continue

            # Query with spinner
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(description="Analyzing with Ollama...", total=None)
                response = rag.query(query)

            # Color-code status
            status_color = {
                "✅ Covered": "green",
                "❌ Not Covered": "red",
                "⚠️ Ambiguous": "yellow",
            }.get(response.status.value, "white")

            console.print(f"\n[bold {status_color}]{response.status.value}[/bold {status_color}]")
            console.print(response.explanation)
            console.print(f"[dim]Reference: {response.reference}[/dim]")
            console.print(f"[dim]Confidence: {response.confidence.value}[/dim]\n")

        except KeyboardInterrupt:
            console.print("\n[cyan]👋 Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
