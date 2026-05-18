<div align="center">

# Synthion

**Turn any lecture into an image and illustrations rich textbook locally for free!**

YouTube playlists, videos, PDFs, audio — in. Chaptered, cited, illustrated HTML textbooks, with real relevant **screenshots** and images from the videos.  
No API keys. No cloud. Small quantized models on your own hardware.

[Installation](#installation) · [Chapter CLI](#chapter-cli) · [Architecture](#how-it-works)

</div>

---

## Why this exists

Large language models can summarize. But summarization is not synthesis. A good textbook has structure: it sequences ideas pedagogically, grounds explanations in source material, places figures where they teach, and cites its origins.

Synthion produces that. It takes a 12-video playlist and outputs a 44-chapter textbook with inline <b>screenshots</b>, footnote citations, and prose that reads like it was written for a student, not scraped from a transcript.

The key insight: **pipeline architecture replaces model scale**. Each stage of the system handles one subproblem — ingestion, planning, retrieval, writing, evaluation — with focused constraints. The result is textbook-quality output from models as small as Qwen 2B, running entirely on consumer Apple Silicon.

## How it works

```
Source (YouTube / video / PDF / text)
        │
        ▼
┌──────────────┐     Transcripts, scene-detected frames,
│    Ingest    │ ──▶ perceptual-deduplicated visuals
└──────┬───────┘
       │
       ▼
┌──────────────┐     Pedagogical outline, chapter plan,
│  Understand  │ ──▶ semantic embedding index
└──────┬───────┘
       │
       ▼
┌──────────────┐     Per-section drafts with scoped retrieval,
│   Compose    │ ──▶ figure anchoring, citation injection
└──────┬───────┘
       │
       ▼
┌──────────────┐     Faithfulness scoring → critique → rewrite
│   Evaluate   │ ──▶ (loop until quality gate passes)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Publish    │ ──▶ HTML  ·  EPUB  ·  Markdown
└──────────────┘
```

Each stage writes durable artifacts to an inspectable workspace. The evaluation loop is what separates this from a one-shot generation — weak chapters get critiqued and rewritten before publication.

**Why the output is good:**
- Section-level RAG — the writer sees only relevant context, not the entire transcript.
- Temporal visual grounding — screenshots are placed by scene alignment, not randomly.
- Explicit quality gate — chapters are scored and rewritten, not just generated once.
- Deterministic workspaces — every intermediate artifact is on disk, inspectable and resumable.

## Output

> 12 videos from Abdul Bari's Dynamic Programming playlist → **44-chapter HTML textbook** on Apple Silicon.

<!-- Add a screenshot of the HTML output here: -->
<!-- ![textbook output](docs/assets/output-preview.png) -->
Checkout example run for DP playlist by Abdul Bari
 [Dynamic Programming introduction.pdf](https://github.com/user-attachments/files/27962006/Dynamic.Programming.introduction.pdf)

```
workspace_v4/runs/<run_id>/
├── output/textbook.html     ← open this
├── output/textbook.epub
├── book/textbook.md
└── run.json
```

---

## Installation

```bash
git clone <repo-url> Synthion && cd Synthion

# Install the package (registers `chapter` and `mte` commands)
pip install --break-system-packages -e .

# Runtime stack
pip install --break-system-packages \
  numpy pillow imagehash sentence-transformers \
  duckduckgo-search faster-whisper mlx-lm \
  pymupdf pdfplumber pdfminer.six pytesseract

# Local inference (Apple Silicon)
CMAKE_ARGS="-DGGML_METAL=on" pip install --break-system-packages llama-cpp-python

# System tools
brew install ffmpeg pandoc tesseract
```

Configure model paths in `config/settings.py`, then verify:

```bash
chapter info
```

---

## Chapter CLI

<img width="1728" height="562" alt="image" src="https://github.com/user-attachments/assets/32aad0d3-07a7-4292-b1a4-c9c131d617d9" />

`chapter` is the primary interface. It builds textbooks, manages a library of past runs, and provides an interactive terminal experience.

### Quick reference

```
chapter                              Interactive menu
chapter build <source> [opts]        Build a textbook
chapter library [--open]             Browse past textbooks
chapter open [run_id]                Open HTML in browser
chapter details [run_id]             Inspect a run
chapter add-chapter <src> --to <id>  Extend an existing textbook
chapter info                         System & model status
```

### `build`

```bash
chapter build <source> [--title TEXT] [--out PATH] [--open]
```
<img width="1032" height="838" alt="image" src="https://github.com/user-attachments/assets/a0957f60-1b55-4de7-a2b8-3d105e68fe7c" />

Build a textbook from any supported source.

```bash
chapter build "https://www.youtube.com/watch?v=ZK3O402wf1c" --open
chapter build lecture.pdf --title "Week 3 Notes"
chapter build ./recording.mp4 --out ./my-book --open
```

### `library`

```bash
chapter library [--open]
```

Lists all textbooks: title, date, chapter count, available formats. Pass `--open` to select one and open it.

### `open`

```bash
chapter open [run_id]
```

Opens the HTML textbook in your default browser. Omit the ID to pick from the library.

### `details`

```bash
chapter details [run_id]
```
<img width="1706" height="828" alt="image" src="https://github.com/user-attachments/assets/cb39458e-44cc-4b64-9119-ccb62d6ee302" />

Shows full metadata: sources processed, chapter titles, output paths, timestamps.

### `add-chapter`

```bash
chapter add-chapter <source> --to <run_id> [--chapter-name TEXT] [--open]
```

Appends new material to an existing textbook without rebuilding from scratch.

```bash
chapter add-chapter "https://youtu.be/newvideo" --to run_20260518T093316Z --open
```

### `info`

```bash
chapter info
```

Displays system readiness: installed tools, model availability, engine version.

### Interactive mode

Running `chapter` without arguments opens a guided menu:

1. **Build** — enter a source URL or path.
2. **Playlist** — paste a YouTube playlist, select videos, choose single-book or per-video mode, group into named chapters.
3. **Library** — browse and open past textbooks.
4. **Info** — verify system status.

The playlist workflow turns an entire course into one structured textbook with chapter-level organization — select which videos to include, name your chapter groups, and build.

---

## License

See `LICENSE`.
