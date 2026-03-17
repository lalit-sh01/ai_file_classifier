# File Classifier

AI-powered file organization using Claude API or self-hosted Ollama models. Classifies documents into categories based on their content.

## Installation

### From Source

```bash
git clone https://github.com/yourusername/file-classifier.git
cd file-classifier
pip install -e .
```

### Requirements

- Python 3.8+
- For PDF/DOCX/XLSX support: Dependencies auto-installed
- For Ollama backend: [Ollama](https://ollama.com) installed and running

## Quick Start

### Using Ollama (Free, Local, Private)

```bash
# Install Ollama
brew install ollama

# Start Ollama server
ollama serve

# Pull a model (one-time)
ollama pull llama3.1:8b

# Check connection
file-classifier --check-ollama

# Dry run first
file-classifier --source ~/Downloads --backend ollama --ollama-model llama3.1:8b --dry-run

# Actually organize
file-classifier --source ~/Downloads --backend ollama --ollama-model llama3.1:8b
```

### Using Claude API (Cloud)

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Dry run
file-classifier --source ~/Downloads --dry-run

# Organize
file-classifier --source ~/Downloads
```

## Categories

```
Finance/
├── Statements/     (bank statements, bills, invoices)
├── Taxes/          (tax returns, receipts, deductions)
└── Investments/    (portfolio, stocks, retirement)

Study/
├── Courses/        (syllabi, assignments, certificates)
├── Notes/          (personal notes, highlights)
└── References/     (manuals, textbooks, papers)

Recreation/
├── Hobbies/        (project plans, guides, collections)
├── Entertainment/  (books, articles, saved content)
└── Travel/         (itineraries, bookings, memories)

Keep/
├── Important/      (IDs, contracts, legal documents)
├── Archives/       (old files to keep)
└── Manuals/        (warranties, instructions)
```

## Usage

```bash
# Basic usage with Ollama
file-classifier -s ~/Downloads --backend ollama --ollama-model llama3.1:8b

# Short alias
fclass -s ~/Downloads -b ~ --backend ollama --dry-run

# Skip folders, only process files
file-classifier -s ~/Downloads --skip-folders

# Custom base directory
file-classifier -s ~/Downloads -b ~/Documents

# Check Ollama status
file-classifier --check-ollama
```

## Options

| Option | Description |
|--------|-------------|
| `-s, --source` | Directory with files to organize (required) |
| `-b, --base-dir` | Where to create categories (default: ~) |
| `--backend` | `claude` or `ollama` (default: claude) |
| `--ollama-model` | Model name for Ollama (default: llama3.1:8b) |
| `--ollama-url` | Ollama API URL (default: http://localhost:11434) |
| `-n, --dry-run` | Preview without moving files |
| `--skip-folders` | Only process files, ignore folders |
| `--check-ollama` | Check Ollama connection and list models |
| `-k, --api-key` | Anthropic API key (for Claude backend) |
| `--batch-size` | Files per API call (default: 10) |

## Supported File Types

- **Text**: `.txt`, `.md`, `.csv`
- **PDF**: `.pdf`
- **Word**: `.docx`
- **Excel**: `.xlsx`, `.xls`

Images and other binary files are automatically skipped.

## Model Recommendations

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `llama3.1:8b` | 4.7 GB | Very Fast | Most users |
| `llama3.2:3b` | 2.0 GB | Extremely Fast | Quick classification |
| `mistral-nemo` | 7.0 GB | Fast | Complex documents |
| `qwen2.5:7b` | 4.6 GB | Fast | Instruction following |

**Recommendation:** Start with `llama3.1:8b` - fast and accurate for file classification.

## How It Works

1. Scans source directory for **files** (top level only)
2. **Folders** are treated as units - classified and moved together
3. Extracts text preview from supported file types
4. Sends batch to AI (Claude or Ollama) for classification
5. Moves files/folders to appropriate subcategory
6. If no subcategory matches, puts in main category folder

## Cost Comparison

| Backend | Cost | Speed | Privacy |
|---------|------|-------|---------|
| Claude API | ~$0.30 per 100 files | Fast | Cloud |
| Ollama | Free | Depends on model | Local |

## Troubleshooting

### Ollama connection refused
```bash
# Make sure Ollama is running
ollama serve
```

### Model not found
```bash
# Pull the model first
ollama pull llama3.1:8b
```

### Out of memory (Ollama)
- Use a smaller model like `llama3.2:3b`
- Reduce `--batch-size` to 5

## License

MIT License - See LICENSE file for details.
