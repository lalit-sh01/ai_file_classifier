#!/usr/bin/env python3
"""Command-line interface for file classifier."""

import argparse
import os
import sys
from pathlib import Path

from .organizer import FileOrganizer
from .ollama_classifier import check_ollama_connection, list_available_models


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered file classifier using Claude or Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Claude API (default)
  python -m file_classifier --source ~/Downloads --dry-run

  # Using Ollama (self-hosted)
  python -m file_classifier --source ~/Downloads --backend ollama --ollama-model llama3.1:8b

  # Check Ollama connection and available models
  python -m file_classifier --check-ollama

  # Skip folders, only process files
  python -m file_classifier --source ~/Downloads --skip-folders
        """
    )

    parser.add_argument(
        "--source", "-s",
        help="Source directory containing files to organize"
    )

    parser.add_argument(
        "--base-dir", "-b",
        default="~",
        help="Base directory for categories (default: ~)"
    )

    parser.add_argument(
        "--api-key", "-k",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (for Claude backend)"
    )

    parser.add_argument(
        "--backend",
        choices=["claude", "ollama"],
        default="claude",
        help="AI backend to use (default: claude)"
    )

    parser.add_argument(
        "--ollama-model",
        default="llama3.1:8b",
        help="Ollama model name (default: llama3.1:8b)"
    )

    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )

    parser.add_argument(
        "--check-ollama",
        action="store_true",
        help="Check Ollama connection and list available models"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files per API call (default: 10)"
    )

    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without moving files"
    )

    parser.add_argument(
        "--skip-folders",
        action="store_true",
        help="Skip folders - only process individual files"
    )

    args = parser.parse_args()

    # Check Ollama connection if requested
    if args.check_ollama:
        print("Checking Ollama connection...")
        if check_ollama_connection(args.ollama_url):
            print(f"✓ Ollama is running at {args.ollama_url}")
            print("\nAvailable models:")
            models = list_available_models(args.ollama_url)
            for model in models:
                print(f"  - {model}")
        else:
            print(f"✗ Cannot connect to Ollama at {args.ollama_url}")
            print("  Make sure Ollama is installed and running:")
            print("    brew install ollama")
            print("    ollama serve")
        return

    # Validate source is provided
    if not args.source:
        print("Error: --source is required (unless using --check-ollama)")
        sys.exit(1)

    # Validate API key for Claude backend
    if args.backend == "claude" and not args.api_key:
        print("Error: API key required for Claude backend. Set ANTHROPIC_API_KEY or use --api-key")
        sys.exit(1)

    # Check Ollama connection if using Ollama
    if args.backend == "ollama":
        if not check_ollama_connection(args.ollama_url):
            print(f"Error: Cannot connect to Ollama at {args.ollama_url}")
            print("Make sure Ollama is installed and running:")
            print("  brew install ollama")
            print("  ollama serve")
            sys.exit(1)
        print(f"✓ Connected to Ollama, using model: {args.ollama_model}")

    # Create organizer
    organizer = FileOrganizer(
        base_dir=args.base_dir,
        api_key=args.api_key,
        dry_run=args.dry_run,
        skip_folders=args.skip_folders,
        backend=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url
    )

    # Ensure directories exist
    organizer.ensure_directories()

    # Process files
    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source directory does not exist: {source}")
        sys.exit(1)

    organizer.process_directory(source, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
