"""Ollama-based classification for self-hosted models."""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import requests


# Category structure
CATEGORIES = {
    "Finance": ["Statements", "Taxes", "Investments"],
    "Study": ["Courses", "Notes", "References"],
    "Recreation": ["Hobbies", "Entertainment", "Travel"],
    "Keep": ["Important", "Archives", "Manuals"]
}


def build_category_prompt() -> str:
    """Build the category list for the AI prompt."""
    lines = [
        "Categories (classify by content, not filename):",
        "",
        "Finance/Statements - Bank statements, credit card bills, invoices, payment receipts",
        "Finance/Taxes - Tax returns, W-2s, 1099s, tax documents, deduction receipts",
        "Finance/Investments - Portfolio statements, stock trades, retirement accounts, brokerage",
        "",
        "Study/Courses - Syllabi, assignments, textbooks, course materials, certifications",
        "Study/Notes - Personal notes, highlights, study guides, lecture notes",
        "Study/References - Manuals, research papers, technical docs, reference materials",
        "",
        "Recreation/Hobbies - Project plans, guides, collections, hobby-related files",
        "Recreation/Entertainment - Books, articles, movies, games, saved content",
        "Recreation/Travel - Itineraries, bookings, tickets, travel memories, trip plans",
        "",
        "Keep/Important - IDs, passports, wills, legal documents, contracts, insurance policies, birth certificates",
        "Keep/Archives - Old files to preserve, completed projects, historical records",
        "Keep/Manuals - Product warranties, instruction guides, appliance manuals, setup guides",
        "",
        "IMPORTANT: Insurance policies, wills, passports, and legal documents go in Keep/Important, NOT Finance."
    ]
    return "\n".join(lines)


def classify_with_ollama(
    items: List[Dict],
    model: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    item_type: str = "files"
) -> List[Dict]:
    """Classify items using Ollama API.

    Args:
        items: List of dicts with item info (files or folders)
        model: Ollama model name to use
        base_url: Ollama API base URL
        item_type: "files" or "folders"

    Returns:
        List of classification results
    """
    categories_text = build_category_prompt()

    if item_type == "folders":
        prompt = f"""Classify each of these {len(items)} folders into one category.
Each folder should be treated as a single unit and moved entirely.

IMPORTANT: Classify based on the FOLDER CONTENTS (sample_files), NOT the folder name.
Folder names may be random or misleading - use the sample files to determine the category.

Categories:
{categories_text}

Respond ONLY with valid JSON in this exact format:
{{"classifications": [{{"index": 0, "category": "Study/Courses", "reason": "brief explanation"}}]}}

Folders to classify:
{json.dumps(items, indent=2)}

JSON response:"""
    else:
        prompt = f"""Classify each of these {len(items)} files into one category.

IMPORTANT: Classify based on the FILE CONTENT (preview text), NOT the filename or file extension.
Filenames may be random or misleading - always use the content preview to determine the category.

Categories:
{categories_text}

Respond ONLY with valid JSON in this exact format:
{{"classifications": [{{"index": 0, "category": "Finance/Taxes", "reason": "brief explanation"}}]}}

Files to classify (focus on the 'preview' content, ignore 'filename'):
{json.dumps(items, indent=2)}

JSON response:"""

    # Call Ollama API
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,  # Low temperature for consistent output
                "num_predict": 2000
            }
        },
        timeout=120
    )

    response.raise_for_status()
    result = response.json()

    # Parse the response
    try:
        content = result.get("response", "")
        # Extract JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        parsed = json.loads(content.strip())
        return parsed.get("classifications", [])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing Ollama response: {e}")
        print(f"Raw response: {result.get('response', 'N/A')[:500]}")
        return []


def validate_category(category: str) -> tuple:
    """Validate and normalize category.

    Returns:
        Tuple of (main_category, sub_category or None)
    """
    category = category.strip()

    if "/" in category:
        parts = category.split("/", 1)
        main = parts[0]
        sub = parts[1] if len(parts) > 1 else None

        # Validate main category exists
        if main not in CATEGORIES:
            return "Keep", "Archives"  # fallback

        # Validate subcategory exists
        if sub and sub not in CATEGORIES[main]:
            return main, None  # use main only

        return main, sub
    else:
        # Just main category
        if category in CATEGORIES:
            return category, None
        return "Keep", "Archives"  # fallback


def check_ollama_connection(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_available_models(base_url: str = "http://localhost:11434") -> List[str]:
    """List available Ollama models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
    except requests.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return []
