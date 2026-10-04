# Architecture decisions: v1 (2025) → v2 (2026) → v2.1 → v2.2

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

v1 always picked something, so silent misfiles were invisible. v2 combines two **independent signals** (embedding rank and LLM choice). When they agree, confidence is high; when they disagree, it is medium. Unreadable files are flagged low. Anything below `min_confidence` lands in **`_Review/`**, where you look once instead of hunting later. *(Superseded in v2.1 by asking: see §17.)* LLM self-reported confidence was deliberately *not* used, because small models are badly calibrated at it.

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
| **Decision** | **The standard library for Office formats.** docx/xlsx/pptx/odt are zip archives of XML; reading paragraphs takes about 40 lines. PDF is the one format that really needs a parser: `pypdf` (pure Python, optional extra), else PyMuPDF if present, else `pdftotext`. Core install: **zero dependencies**. *(v2.1 makes pypdf required: see §15.)* |

## 11. Images and scans

v1 skipped them. In 2026, 4B vision models (gemma3:4b, qwen2.5vl:3b) run on a laptop. v2 sends images to the model when `vision = true`; otherwise they're sorted by filename and sent to review. OCR'ing scanned PDFs through the vision model is the natural next step. *(Done in v2.1: see §15.)*

## 12. Runtime coupling

| | |
|---|---|
| **v1** | Ollama's `/api/generate` or the Anthropic SDK, with duplicated code paths |
| **Alternatives** | Embed llama.cpp in-process (llama-cpp-python, MLX) · **talk to a local server** |
| **Decision** | **A local server, through a three-method interface** (`choose`, `embed`, `status`). Ollama natively; *anything* OpenAI-compatible (LM Studio, llama.cpp, vLLM, Jan, mlx-lm) via one adapter; Claude as an opt-in cloud fallback *(removed in v2.1: see §16)*. Embedding the runtime would mean shipping GPU builds per platform, which is exactly the problem Ollama and LM Studio already solve. HTTP uses `urllib`, so there are no SDKs. |

## 13. Default model

v1 defaulted to the **cloud** (Claude) and suggested llama3.1:8b locally. v2 defaults to **local**: `qwen3:4b` plus `embeddinggemma` (about 3 GB together, comfortable on 8 GB machines). `fclass doctor` suggests alternatives for your RAM.

## 14. Kept from v1

- **Top level only, folders as units.** Rearranging the insides of someone's project folder is never what they want.
- **Content over filename.** It's in the prompt, and the benchmark uses meaningless filenames to enforce it.
- **The 4 × 3 taxonomy** as the default, now just a starting point.

---

# v2.1: offline, any file type, asks when in doubt

The brief was sharpened to: *an offline, on-device, intelligent file organiser that is customisable and asks when in doubt*, for **any** file type. Each part of that sentence forced a decision.

## 15. Any file type, recognised by content

| | |
|---|---|
| **v2** | Dispatched on the file extension; images, media, archives and installers were skipped by default rules |
| **Alternatives** | Extension table · `libmagic`/`python-magic` (a native library) · **magic-byte sniffing in Python** |
| **Decision** | **Sniff the first bytes.** About 25 signatures cover PDF, the zip family (then told apart by their members: Word, Excel, PowerPoint, OpenDocument, EPUB, Android apps, plain archives), images, HEIC, audio, video, tar/gz, 7z/rar, RTF, SQLite, Windows/macOS/Linux programs, installers, disk images and fonts. Text is then split into HTML, email and notebooks. A PDF named `download` or a PNG named `.txt` is read correctly. **Every file gets a preview**: at minimum a `Type: … Size: … Modified: …` line plus whatever is readable (archive listings, MP3 tags, printable strings from binaries). The default skip rules were removed; rules remain as an opt-in. |
| **PDFs** | `pypdf` becomes the one required dependency (pure Python, no native code). Probing showed `pypdf` can't extract images without Pillow, *but* a scanned page's `/DCTDecode` stream **is already a JPEG file**, so its raw bytes go straight to the vision model. Raw-pixel (`/FlateDecode`) pages are wrapped into a PNG with `zlib` + `struct` (about 15 lines). JBIG2/CCITT fax scans fall back to `pdftoppm` when installed, otherwise to a question. |
| **Pictures** | Photos, screenshots and scanned pages go to a **separate vision model** (`vision_model = "gemma3:4b"`, used automatically when installed) while documents keep the stronger text model. Embeddings can't see pixels, so pictures skip the fast pass. |

## 16. Offline is enforced, not promised

| | |
|---|---|
| **v2** | Local by default, Claude opt-in |
| **Decision** | **The cloud backend is gone, and the URL is checked.** Every model URL is classified as *this computer* (loopback), *your network* (private ranges, `.local`) or *the internet*; the internet is refused unless `allow_remote = true`. `fclass doctor` states which one applies. Your own NAS or desktop GPU box counts as "your devices"; a public API does not. |

## 17. Ask, don't guess

| | |
|---|---|
| **v2** | Unsure files moved to `_Review/` |
| **Problem** | Moving a file you'll have to move again is still a guess, just a labelled one, and `_Review/` becomes another pile |
| **Decision** | **`when_unsure = "ask"` (default): an unsure file stays where it is until you answer.** At the terminal, `sort` asks before it moves anything. Otherwise (`-y`, `watch`) the question is queued for `fclass ask`. A question shows the file's detected type, the model's reason and its top three guesses; you can pick one, see all categories, **create a new category on the spot**, or leave it. Every answer becomes an example (§9), so the same doubt comes up less. `review_folder` and `leave` remain as options. |
| **Learning closes the loop** | Unreadable files (installers, binaries) used to stay at low confidence forever, so the promise "answers are remembered" was false for them. Now, when such a file closely resembles one you sorted (cosine above τ) **and** the model agrees, it is filed with high confidence. Verified with real models: after one answer filing a Zoom installer under a new `Software/Installers`, a Teams installer arriving in `watch` was filed there on its own. |
| **Also** | One slow file (for example a large photo on a CPU-only machine) now becomes a question instead of aborting the run; only an unreachable server stops it. Vision calls get at least 5 minutes. |

## 18. Customisation without hand-editing

`fclass categories add/remove`, new categories created while answering, and `discover` all edit the user's TOML **in place**. Comments and layout are preserved, the new file must parse before it is written, and the previous one is kept as `config.toml.bak`. (A first draft of the block parser would have deleted the comment heading that follows the last category. Caught in review before it shipped, and pinned by a test.)

## 19. `fclass watch`: sorting downloads as they land

| | |
|---|---|
| **Alternatives** | OS file events (`watchdog`, FSEvents, inotify; a native dependency, per-platform quirks) · **polling** |
| **Decision** | **Poll every 5 s, standard library only.** Downloads arrive a few at a time, so a 5-second poll costs nothing and behaves the same on macOS, Linux and Windows. An item is touched only after it has been **unchanged for `settle_seconds`**, and browser partial files (`*.crdownload`, `*.part`, `*.download`) are ignored, so a half-finished download is never moved. Only new arrivals are handled unless `--existing`. Confident items are filed into a daily journal (`fclass undo --last 1` reverts the latest); unsure ones are queued, with a desktop notification (`osascript` / `notify-send`, best effort). `--print-service` emits a launchd or systemd unit to run it at login. |

## 20. `fclass discover`: categories from your own folder

| | |
|---|---|
| **Approach** | Embed every file, group similar ones, have the LLM name each group (an existing category or a new `Area/Topic` with a description), and merge groups given the same name. Nothing is written until you accept, and the three most central files of each accepted group become examples. |
| **What failed** | **Choosing the number of groups by silhouette score** picked 4 groups for 12 real categories (27% purity): silhouette scores were about 0.06, meaning documents this varied barely form natural clusters. **Per-file LLM captions** ("bank statement", "user manual") were accurate but clustered no better than raw content (63% vs 57–65%) at 7 s per file, so they were dropped. (A single-field caption schema produced garbage like `text\|json\|markdown`; a describe-then-label schema fixed it. The same reason-first lesson as §3.) |
| **What worked** | **Over-split, then let meaning merge.** About 3 files per group gives 75% purity before naming (vs 57% at 4 and 53% at 5); the LLM gives near-duplicate groups the same name, and those merge. End to end, from no categories: **12 categories in 212 s**; about two-thirds of grouped files sit with their kind, and 8 of 60 files did not group. Clean groups (assignments, manuals, identity documents, tax forms) next to a grab-bag "Study/Notes" and bank statements split in two. A good first draft to edit, not a finished taxonomy. |
| **Caveat** | "Purity" against a reference taxonomy undersells it: a proposal that splits *Keep/Important* into *Identity* and *Contracts* is reasonable but scores as wrong. That's why the result is a proposal you edit, not a decision. |

---

# v2.2: Mac first, ready for other people

The target became an **open-source utility that a Mac user installs and trusts in minutes**. That changed what mattered: real Mac behaviour, signals the OS already has, and a way for anyone to measure it on their own machine.

## 21. Use what the operating system already knows

| | |
|---|---|
| **Download origin** | macOS records where every download came from (`kMDItemWhereFroms`, a binary plist in an extended attribute); Chrome and Firefox on Linux write `user.xdg.origin.url`. A PDF from `netbanking.hdfcbank.com` is a bank document whatever its name is. The host and path are added to each preview; **query strings are dropped**, since they carry session tokens, not meaning. Python has no `os.getxattr` on macOS, so it is read through libc with `ctypes`, with the built-in `xattr` tool as a fallback. |
| **Pictures** | macOS ships `sips`, which converts HEIC/AVIF (iPhone photos, the most common Mac file v2.1 could not read) to JPEG and shrinks large images to 1600 px before the vision model reads them. No Pillow, no dependency. |
| **Rejected** | Spotlight (`mdls`) for the same attributes: it depends on indexing state, while the attribute is on the file itself. |

## 22. Questions without a terminal

| | |
|---|---|
| **Problem** | A file organiser that only asks in a terminal won't get its questions answered by most Mac users |
| **Alternatives** | Notification action buttons (need a signed app bundle on current macOS) · a menu-bar app (PyObjC/rumps, a heavy dependency) · **AppleScript's built-in pickers** |
| **Decision** | **`osascript` dialogs**: `choose from list` (guesses first, then every category, *New category…*, *Leave it where it is*) and `display dialog` for new names. Zero dependencies, native look. `fclass ask --dialog` answers saved questions; `[watch] ask_with = "dialog"` asks right away and, unanswered after two minutes, saves the question instead of blocking. A menu-bar app stays on the roadmap for v3. |

## 23. Measured on your machine, tested on a real Mac

- **`fclass bench`** ships the benchmark inside the package and measures accuracy and speed **end to end** with the user's own models, in a throwaway state folder so their cache and examples are neither used nor touched. It prints a table to share, which is how Apple Silicon numbers will reach the README.
- **CI runs on macOS.** This project is developed without a Mac, so GitHub's macOS runner executes the macOS-only tests against the real `sips`, `xattr` and `osacompile` (the AppleScript is compiled, not shown). Linux runs on Python 3.11 and 3.13.

## Not done (yet)

- v2.3: proposing clear names for files with unhelpful ones (`scan_0042.pdf` → `2025-03 HDFC Bank Statement.pdf`), applied only when approved
- v2.4: `fclass setup` (guided first run), Homebrew tap and PyPI release, contributing guide
- v3.0: `fclass find` (search by meaning), duplicates, a menu-bar app
- JBIG2 scans without `pdftoppm`; video and audio content beyond MP3 tags
