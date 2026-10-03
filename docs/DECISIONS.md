# Architecture decisions: v1 (2025) → v2 (2026)

Every v1 choice was put back on the table. Where the answer wasn't obvious, it
was measured. All numbers come from `benchmarks/` (60 realistic documents,
12 categories, deliberately generic filenames like `scan_0042.pdf`, run on a
4-core CPU with no GPU, so treat absolute latencies as worst case).

## Scoreboard

| v1 choice | v2 choice | Measured effect |
|---|---|---|
| LLM classifies everything | Embeddings first, LLM for close calls | 91.7% at 7.5 s/file vs 93.3% at 22 s/file |
| 10 files per prompt | 1 file per call | **76.7% → 93.3%** with the same model |
| "Respond with JSON" + string parsing | Enum-constrained decoding | 11/60 unusable answers (1.7B, batched) → 0 |
| Hard-coded categories ×2 | One TOML file | — |
| Move immediately | Plan → confirm → journal → undo | — |
| Always guess | `_Review/` below a confidence floor | — |
| No memory | Corrections become examples | Never lowers accuracy (tested); helps most on near-duplicates |
| PyMuPDF + pandas + docx + openpyxl | stdlib zip/XML, optional pypdf | ~100 MB → 0 required dependencies |
| Cloud default | Local default (`qwen3:4b` + `embeddinggemma`) | about 3 GB of models |

---

## 1. What does the classifying?  *The big one*

| | |
|---|---|
| **v1** | A generative LLM for every file |
| **Alternatives** | (a) LLM for every file · (b) embeddings only (cosine similarity to category descriptions) · (c) zero-shot NLI models · (d) a fine-tuned classifier · (e) **hybrid**: embeddings first, LLM only for close calls |
| **Evidence** | Embeddings alone (`embeddinggemma`, 300M params) reached **80%** at **0.1 s/file**, while `qwen3:1.7b` reached only 50% at ~5 s. The embedder's misses were *semantic*: it filed leases, insurance policies and offer letters under Finance because they mention money. `qwen3:4b` reading every file reached **93%** but costs seconds per file. The margin sweep showed that when embeddings are *confident* (margin ≥ 0.04) they are right **95%** of the time, which covers about two thirds of files. Hybrid result: **91.7% at 7.5 s/file**. A first version gave the LLM only the embedding top-3 and scored 86.7%: on close calls the true answer is outside the top 3 about 30% of the time, so the LLM now sees every category, best guesses first. |
| **Decision** | **(e) Hybrid.** Embeddings settle the clear majority almost for free; the LLM spends its time only where judgement is needed. (d) was rejected because a fine-tuned classifier freezes the taxonomy, which defeats "customisable". (c) NLI models are slower than embeddings and weaker than modern 4B LLMs. |

## 2. Batch many files per prompt, or one file per call?

| | |
|---|---|
| **v1** | 10 files per prompt, answers matched back by index |
| **Why it seemed right** | Fewer calls, and cheaper with a cloud API |
| **What's wrong locally** | There is no per-call fee locally, but long multi-answer generations are slow on small hardware. Index mapping is fragile: a dropped or duplicated index silently misfiles a document. Attention is shared across 10 documents, so they bleed into each other. |
| **Evidence** | Same model (qwen3:4b): v1-batch **76.7%** at 15 s/file vs single-file **93.3%**. With qwen3:1.7b, v1-batch returned **11 of 60 answers unusable**. |
| **Decision** | **One file per call.** The system prompt is identical across calls, so the runtime's prompt cache absorbs most of the repeated cost. |

## 3. How is the model kept to valid answers?

| | |
|---|---|
| **v1** | Asked nicely for JSON, then split strings on `` ``` `` and fell back to `Keep/Archives` |
| **Alternatives** | Free-text + regex · `format: "json"` · **grammar/JSON-schema-constrained decoding** · tool calling |
| **Decision** | **Schema-constrained decoding with the category as an `enum`.** Ollama, llama.cpp, LM Studio and vLLM all support it in 2026, so the model physically cannot emit a category that doesn't exist. `reason` is generated *before* `category`, which buys a sentence of reasoning without enabling slow "thinking" mode. For Claude, forced tool use does the same job. Result: **0 invalid answers** across every single-file run. A tempting tweak, asking for `reason` as a terse noun phrase to tidy the display, made qwen3:4b write the category name there and re-file a McKinsey report as a legal document. The sentence *is* the reasoning, so it stays, and the display trims it instead. |

## 4. Thinking mode on or off?

Qwen3 models "think" by default, producing hundreds of tokens of reasoning first. For a 12-way choice with a written reason, that is 3–10× slower for no measurable gain, so v2 sends `think: false` to thinking-capable models.

## 5. Where do categories live?

| | |
|---|---|
| **v1** | Hard-coded in two Python modules (duplicated, and already drifting apart) |
| **Alternatives** | Python constants · YAML · **TOML** · a GUI · letting the LLM invent the taxonomy |
| **Decision** | **One TOML file** (`fclass init`). Python 3.11 reads TOML natively, so no dependency is needed, and it's comment-friendly for humans. Each category is a *path* plus a *plain-language description*, and the description is what both models read. Editing it automatically invalidates the cache via a config fingerprint. Taxonomy *discovery* (the LLM proposes folders from your files) is a promising next step, kept out of scope here. |

## 6. Deterministic rules

v1 skipped files by a hard-coded extension list. v2 makes **rules first-class config**: glob → category, or glob → skip. Rules are free, instant and 100% predictable, so use them for anything you can describe by name.

## 7. Moving files: immediate, or plan → apply?

| | |
|---|---|
| **v1** | Classified and moved in the same loop; `--dry-run` was the only safety |
| **Alternatives** | Move immediately · **plan file + confirm + journal + undo** · tag/symlink instead of moving (macOS tags, xattrs) |
| **Decision** | **Plan → show → confirm → apply, with a journal and `fclass undo`.** Each move is flushed to the journal before the next starts, so undo survives a crash. Collisions never overwrite. Tagging was rejected because it isn't portable and most file views ignore it; moving matches the original vision. |

## 8. What happens when the model isn't sure?

v1 always picked something, so silent misfiles were invisible. v2 combines two **independent signals** (embedding rank and LLM choice). When they agree, confidence is high; when they disagree, it is medium. Unreadable files are flagged low. Anything below `min_confidence` lands in **`_Review/`**, where you look once instead of hunting later. LLM self-reported confidence was deliberately *not* used, because small models are badly calibrated at it.

## 9. Learning from you

| | |
|---|---|
| **v1** | None. The same mistake every run |
| **Alternatives** | Fine-tuning/LoRA · editing descriptions · **storing corrections as examples** |
| **Decision** | **Examples, used twice.** Corrections made during review (or via `fclass teach`) are saved. (1) **The LLM** sees the most similar ones as few-shot precedent, which carries the subtle preferences ("my relieving letters are archives"). (2) **The embedding ranker** gets extra evidence for a category *only* when a file closely resembles one of its examples (cosine above τ). |
| **What benchmarking caught** | The first version z-normalised example similarity across categories. With examples in only *one* category, that tilted **every** file towards it. A real-model run surfaced the bug, a "teach a single category" scenario now guards it, and τ and the weight were tuned so that teaching one category never lowers accuracy elsewhere. On this dataset (no near-duplicates) the embedding-side gain is about 0; it pays off on recurring documents like monthly statements from the same bank. |

## 10. Reading documents

| | |
|---|---|
| **v1** | PyMuPDF + python-docx + pandas + openpyxl, about 100 MB of dependencies to read a preview |
| **Decision** | **The standard library for Office formats.** docx/xlsx/pptx/odt are zip archives of XML; reading paragraphs takes about 40 lines. PDF is the one format that really needs a parser: `pypdf` (pure Python, optional extra), else PyMuPDF if present, else `pdftotext`. Core install: **zero dependencies**. |

## 11. Images and scans

v1 skipped them. In 2026, 4B vision models (gemma3:4b, qwen2.5vl:3b) run on a laptop. v2 sends images to the model when `vision = true`; otherwise they're sorted by filename and sent to review. OCR'ing scanned PDFs through the vision model is the natural next step.

## 12. Runtime coupling

| | |
|---|---|
| **v1** | Ollama's `/api/generate` or the Anthropic SDK, with duplicated code paths |
| **Alternatives** | Embed llama.cpp in-process (llama-cpp-python, MLX) · **talk to a local server** |
| **Decision** | **A local server, through a three-method interface** (`choose`, `embed`, `status`). Ollama natively; *anything* OpenAI-compatible (LM Studio, llama.cpp, vLLM, Jan, mlx-lm) via one adapter; Claude as an opt-in cloud fallback. Embedding the runtime would mean shipping GPU builds per platform, which is exactly the problem Ollama and LM Studio already solve. HTTP uses `urllib`, so there are no SDKs. |

## 13. Default model

v1 defaulted to the **cloud** (Claude) and suggested llama3.1:8b locally. v2 defaults to **local**: `qwen3:4b` plus `embeddinggemma` (about 3 GB together, comfortable on 8 GB machines). `fclass doctor` suggests alternatives for your RAM.

## 14. Kept from v1

- **Top level only, folders as units.** Rearranging the insides of someone's project folder is never what they want.
- **Content over filename.** It's in the prompt, and the benchmark uses meaningless filenames to enforce it.
- **The 4 × 3 taxonomy** as the default, now just a starting point.

## Not done (yet)

- `fclass watch` to sort new downloads as they land
- Taxonomy discovery from an existing messy folder
- Vision OCR for scanned PDFs
- A menu-bar/tray app wrapper
