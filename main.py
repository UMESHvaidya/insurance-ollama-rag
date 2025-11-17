"""Main entry point for the Insurance Policy RAG system with Ollama"""

import sys
from pathlib import Path
from dotenv import load_dotenv

from insurance_rag import InsurancePolicyRAG
from insurance_rag.config import get_settings


def check_ollama():
    """Check if Ollama is running"""
    import ollama

    try:
        ollama.list()
        return True
    except:
        return False


import argparse
from insurance_rag.app.app import main as run_web_app


def run_cli():
    """Run example queries on a policy document"""
    # Load environment variables
    load_dotenv()

    print("=" * 70)
    print("🏥 INSURANCE POLICY RAG SYSTEM (Local Ollama + Free)")
    print("=" * 70)
    print()

    # Check Ollama
    if not check_ollama():
        print("❌ Ollama is not running!")
        print()
        print("Please start Ollama:")
        print("  1. Install: https://ollama.com/download")
        print("  2. Start: ollama serve")
        print("  3. Pull model: ollama pull gemma2:2b")
        print()
        return 1

    # Get settings
    try:
        settings = get_settings()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return 1

    # Initialize RAG system
    try:
        rag = InsurancePolicyRAG()
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print()
        print("Make sure Ollama model is available:")
        print(f"  ollama pull {settings.ollama_model}")
        return 1

    # Check for policy document
    pdf_path = settings.data_dir / "insurance_policy.pdf"

    if not pdf_path.exists():
        print(f"❌ Policy document not found: {pdf_path}")
        print(f"   Please place your insurance policy PDF in: {settings.data_dir}/")
        return 1

    # Load policy
    try:
        rag.load_policy(pdf_path)
    except Exception as e:
        print(f"❌ Error loading policy: {e}")
        return 1

    # Example queries
    queries = [
        "Is cataract surgery covered?",
        "What about dental treatment?",
        "Are pre-existing conditions covered?",
        "Is ambulance service included?",
        "Is piles surgery covered?",
        "Is Hair transplant covered?",
    ]

    print("📋 Running example queries...")
    print("=" * 70)

    for query in queries:
        print(f"\n❓ QUERY: {query}")
        print("-" * 70)

        try:
            response = rag.query(query)
            print(response)
        except Exception as e:
            print(f"❌ Error: {e}")

        print()

    print("=" * 70)
    print("✅ Analysis complete!")
    print("   For interactive mode, run: poetry run python scripts/interactive.py")

    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Insurance Policy RAG System")
    parser.add_argument(
        "mode",
        nargs="?",
        default="cli",
        choices=["cli", "web"],
        help="Mode to run the application in (default: cli)",
    )
    args = parser.parse_args()

    if args.mode == "web":
        print("🚀 Starting web application...")
        run_web_app()
        return 0
    else:
        return run_cli()


if __name__ == "__main__":
    sys.exit(main())
