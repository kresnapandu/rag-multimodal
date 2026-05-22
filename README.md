## Multimodal RAG

A production-ready **Retrieval-Augmented Generation** system optimised for **telecommunications hardware datasheets** (BTS, RRU, antennas, optical transceivers, routers, switches).

Handles PDF documents with mixed content — plain text, specification tables, and embedded diagrams — and answers technical questions with precise spec values.

---

## Key Features

| Feature | Implementation |
|---|---|
| **Multimodal ingestion** | Unstructured.io extracts text + HTML tables + base64 images from PDFs |
| **AI enriched summaries** | GPT-4o Vision generates searchable descriptions of table/image content |
| **Semantic chunking** | LangChain `SemanticChunker` splits on embedding-space sentence boundaries |
| **History-aware retrieval** | Query reformulation via chat history — standalone, searchable rewrites |
| **Token minimisation** | Sentence-level keyword highlighting (60–80 % fewer context tokens) |
| **Score-based filtering** | Only chunks above a cosine similarity threshold reach the LLM |
| **History compression** | Older turns auto-summarised after N exchanges — constant memory overhead |
| **Telecom metadata** | Regex extracts freq bands, power specs, data rates, standards as ChromaDB fields |
| **Bilingual** | Detects Indonesian / English automatically; answers in the same language |
| **Metrics & eval** | Latency, token usage, hit@k, MRR exported to CSV |

---

## Token Optimisation — How It Works

```
Raw retrieval (naive):   5 chunks × 1 000 chars = 5 000 chars context
After highlighting:      5 chunks × 200  chars = 1 000 chars context  ← 80% saved
```

**Step-by-step pipeline per query:**

```
User query
   │
   ├─ 1. Language detection (zero-cost heuristic)
   ├─ 2. Query reformulation  (uses chat history → self-contained query)
   ├─ 3. ChromaDB similarity search  (top-K candidates)
   ├─ 4. Score gate  (drop chunks below threshold)
   ├─ 5. Deduplication  (remove near-identical chunks)
   ├─ 6. Sentence highlighting  (keyword overlap → top-N sentences per chunk)
   ├─ 7. Char cap  (hard limit per chunk)
   └─ 8. LLM answer generation  (grounded, technical, bilingual)
```

**History compression:** After `MAX_HISTORY_TURNS` Q/A pairs, all but the last 2 exchanges are summarised into a single ~100-word SystemMessage. This keeps history tokens roughly constant regardless of session length.

---

## Architecture

```
telecom-rag/
├── main.py                   CLI entry point & command dispatcher
├── config.py                 All tuneable parameters (env-override support)
│
├── rag/
│   ├── ingestion/
│   │   ├── loader.py         Unstructured partitioning + payload extraction
│   │   ├── chunker.py        SemanticChunker + telecom metadata regex
│   │   ├── vectorstore.py    ChromaDB CRUD wrapper
│   │   └── vision.py         GPT-4o multimodal summary generator
│   │
│   ├── retrieval/
│   │   └── retriever.py      Score gate → dedupe → highlight → cap pipeline
│   │
│   ├── generation/
│   │   ├── history.py        Chat history with auto-compression
│   │   └── generator.py      History-aware query rewrite + answer generation
│   │
│   └── utils/
│       ├── language.py       Heuristic Indonesian/English detector
│       ├── highlight.py      Sentence-level keyword overlap scorer
│       └── metrics.py        In-memory metrics store + CSV export
│
├── docs/                     Drop your PDF datasheets here
├── examples/
│   └── eval_example.jsonl    Sample offline evaluation set
├── .env.example
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Prerequisites (system dependencies)

```bash
# Ubuntu / Debian
sudo apt-get install poppler-utils tesseract-ocr libmagic-dev

# macOS
brew install poppler tesseract libmagic

# Windows — install via the Windows installer:
#   Poppler:   https://github.com/oschwartz10612/poppler-windows/releases
#   Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
#   libmagic:  installed automatically via python-magic-bin (pip)
```

### 2. Install Python dependencies

```bash
git clone https://github.com/your-username/telecom-rag.git
cd telecom-rag
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 4. Run

```bash
python main.py
```

---

## CLI Commands

```
> /ingest docs/                         Ingest all files in the docs folder
> /ingest docs/rru_4255_datasheet.pdf   Ingest a single file
> /count                                Show how many chunks are in the DB

> What is the maximum transmit power?   Plain question (uses default filter)
> Berapa gain antena pada 1800 MHz?     Indonesian question — auto-detected

> /where has_specs=true                 Only search chunks that have spec values
> /where is_summary=true                Only search AI-enriched summary chunks
> /clear_where                          Remove the filter

> /ask where:has_specs=true What EIRP does the antenna achieve at 2100 MHz?

> /history                              Show current turn count
> /history clear                        Clear conversation history

> /eval examples/eval_example.jsonl     Run hit@k + MRR offline evaluation
> /export metrics/                      Export all metrics to CSV files

> quit
```

---

## Configuration

All parameters can be overridden via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `CHAT_MODEL` | `gpt-4o` | LLM for answer generation and vision |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `RETRIEVAL_K` | `8` | Candidate chunks fetched from ChromaDB |
| `SCORE_THRESHOLD` | `0.35` | Min cosine similarity to pass the score gate |
| `MAX_CONTEXT_DOCS` | `5` | Max chunks sent to the LLM per query |
| `HIGHLIGHT_SENTENCES` | `5` | Sentences extracted per chunk via keyword overlap |
| `MAX_CHUNK_CHARS` | `1200` | Hard character cap per chunk after highlighting |
| `MAX_HISTORY_TURNS` | `6` | Compress history after this many Q/A pairs |
| `RAG_COLLECTION` | `telecom_rag` | ChromaDB collection name |

**Tuning tips for telecom datasheets:**
- Raise `SCORE_THRESHOLD` to `0.45–0.55` for high-precision spec lookups
- Lower it to `0.25` if documents are very technical and jargon-heavy
- Increase `HIGHLIGHT_SENTENCES` to `8` for long spec tables
- Set `MAX_CONTEXT_DOCS=3` to reduce cost on simple queries

---

## Offline Evaluation

Create a JSONL file with one query per line:

```json
{"question": "What is the max output power in dBm?", "gold_sources": ["rru_datasheet.pdf"], "k": 5}
```

Run:

```
> /eval examples/eval_example.jsonl
  hit=1 rank=1 rr=1.000 | What is the max output power in dBm?
  ...
  hit_rate=0.8750  MRR=0.7083  n=8
```

Export all metrics including eval results:

```
> /export metrics/
  Exported: metrics/ingestion.csv
  Exported: metrics/query.csv
  Exported: metrics/eval.csv
```

---

## Telecom Metadata Filtering

During ingestion, the system auto-extracts structured metadata per chunk:

| Field | Example values |
|---|---|
| `freq_bands` | `["700 MHz", "2100 MHz", "28 GHz"]` |
| `power_specs` | `["43 dBm", "40 W", "20 dBi"]` |
| `data_rates` | `["25 Gbps", "10 Gbps"]` |
| `standards` | `["5G NR", "LTE", "CPRI", "ECPRI"]` |
| `has_specs` | `true` / `false` |
| `is_summary` | `true` for AI-enriched chunks |
| `chunk_type` | `semantic_text` / `multimodal_summary` |

Use `/where` to filter by any of these before asking:

```
/where has_specs=true
/ask where:is_summary=true What interfaces does the O-RU support?
```

---

## License

MIT
