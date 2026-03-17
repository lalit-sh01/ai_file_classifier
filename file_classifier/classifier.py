"""AI classification using Claude API."""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import anthropic


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


def classify_bulk(
    files_info: List[Dict],
    api_key: Optional[str] = None
) -> List[Dict]:
    """Classify multiple files using Claude API.

    Args:
        files_info: List of dicts with 'index', 'filename', 'preview'
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)

    Returns:
        List of classification results
    """
    client = anthropic.Anthropic(
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
    )

    categories_text = build_category_prompt()

    prompt = f"""Classify each of these {len(files_info)} files into one of these categories:

IMPORTANT: Classify based on the FILE CONTENT (preview text), NOT the filename or file extension.
Filenames may be random or misleading - always use the content preview to determine the category.

{categories_text}

Respond ONLY with valid JSON in this exact format:
{{"classifications": [{{"index": 0, "category": "Finance/Taxes", "reason": "brief explanation"}}]}}

Files to classify (focus on the 'preview' content, ignore 'filename'):
{json.dumps(files_info, indent=2)}

JSON response:"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    return parse_classification_response(response.content[0].text)


def parse_classification_response(text: str) -> List[Dict]:
    """Parse JSON response from Claude."""
    try:
        # Extract JSON if wrapped in markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        result = json.loads(text.strip())
        return result.get("classifications", [])
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {text[:500]}")
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
