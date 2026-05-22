"""
Telecom Multimodal RAG — interactive CLI.

Run:
    python main.py

Commands
--------
  /ingest <path>                 Ingest a file or folder (recursive)
  /count                         Show vector DB item count
  /where <key=val ...>           Set a persistent metadata filter (AND)
      Examples:
        /where is_summary=true
        /where has_specs=true
        /where standards~5G NR   (substring match on standards field)
  /clear_where                   Clear the persistent filter
  /ask where:<key=val> <q>       One-off filtered question
  /history clear                 Clear conversation history
  /export [dir]                  Export metrics CSV to ./metrics (or given dir)
  /eval <eval.jsonl>             Offline hit@k / MRR evaluation
  quit / exit                    Exit
"""

import json
import os
import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import Config
from rag.generation.generator import RAGGenerator
from rag.generation.history import HistoryManager
from rag.ingestion.chunker import TelecomChunker
from rag.ingestion.loader import extract_payload, partition_file
from rag.ingestion.vectorstore import VectorStore
from rag.ingestion.vision import create_multimodal_summary
from rag.retrieval.retriever import TelecomRetriever
from rag.utils.metrics import MetricsStore, Timer

load_dotenv()


# ── App bootstrap ──────────────────────────────────────────────────────────────

def build_app(cfg: Config):
    embeddings = OpenAIEmbeddings(model=cfg.embed_model)
    model = ChatOpenAI(model=cfg.chat_model, temperature=cfg.temperature)

    vectorstore = VectorStore(cfg, embeddings)
    chunker = TelecomChunker(embeddings, cfg)
    retriever = TelecomRetriever(vectorstore, cfg)
    history = HistoryManager(model, max_turns=cfg.max_history_turns)
    generator = RAGGenerator(model, retriever, history, cfg)
    metrics = MetricsStore()
    return model, vectorstore, chunker, generator, retriever, metrics


# ── Ingestion ──────────────────────────────────────────────────────────────────

def ingest(paths: list, chunker, vectorstore, model, metrics: MetricsStore):
    for path in paths:
        if not os.path.exists(path):
            print(f"  [WARN] Not found: {path}")
            continue
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    _ingest_file(os.path.join(root, fn), chunker, vectorstore, model, metrics)
        else:
            _ingest_file(path, chunker, vectorstore, model, metrics)


def _ingest_file(path, chunker, vectorstore, model, metrics: MetricsStore):
    t = Timer()
    print(f"  Ingesting: {path}")
    try:
        elements, mime = partition_file(path)
        payload = extract_payload(elements)

        def summary_fn(text, tables, images):
            return create_multimodal_summary(model, text, tables, images)

        docs = chunker.build_documents(path, payload, mime, ai_summary_fn=summary_fn)
        vectorstore.add_documents(docs)

        elapsed = t.elapsed_ms()
        print(
            f"    OK  {os.path.basename(path)} | "
            f"{len(docs)} chunks | tables={len(payload['tables'])} "
            f"images={len(payload['images'])} | {elapsed} ms"
        )
        metrics.log_ingestion(
            path=path,
            file=os.path.basename(path),
            mime=mime,
            docs=len(docs),
            tables=len(payload["tables"]),
            images=len(payload["images"]),
            latency_ms=elapsed,
            status="ok",
        )
    except Exception as exc:
        elapsed = t.elapsed_ms()
        print(f"    ERROR: {exc}")
        metrics.log_ingestion(
            path=path,
            file=os.path.basename(path),
            latency_ms=elapsed,
            status="error",
            error=str(exc),
        )


# ── Metadata filter parsing ────────────────────────────────────────────────────

def parse_where(s: str) -> dict:
    """
    Parse 'key=value key2=value2' into a ChromaDB-compatible filter dict.
    Supports bool ('true'/'false') and int coercion.
    Substring filters (key~substr) are stored with a __contains__ prefix
    and applied as post-filters in the retriever (ChromaDB limitation).
    """
    result = {}
    for tok in s.strip().split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            v = v.strip()
            if v.lower() == "true":
                result[k] = True
            elif v.lower() == "false":
                result[k] = False
            elif v.isdigit():
                result[k] = int(v)
            else:
                result[k] = v
        elif "~" in tok:
            k, v = tok.split("~", 1)
            result[f"__contains__:{k}"] = v
    return result


# ── Offline evaluation ─────────────────────────────────────────────────────────

def run_eval(eval_path: str, retriever: TelecomRetriever, metrics: MetricsStore):
    if not os.path.exists(eval_path):
        print(f"  [ERROR] Eval file not found: {eval_path}")
        return

    rows = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("  Eval set is empty.")
        return

    hits, mrr_sum = 0, 0.0
    for item in rows:
        q = item.get("question", "")
        gold = {g.lower() for g in item.get("gold_sources", [])}
        k = int(item.get("k", 5))

        retrieved = retriever.eval_retrieve(q, k=k)
        found_rank = next(
            (i for i, (src, _) in enumerate(retrieved, 1) if src.lower() in gold),
            None,
        )
        hit = 1 if found_rank else 0
        rr = (1.0 / found_rank) if found_rank else 0.0
        hits += hit
        mrr_sum += rr

        print(f"  hit={hit} rank={found_rank} rr={rr:.3f} | {q[:70]}")
        metrics.log_eval(
            question=q, k=k,
            gold=json.dumps(list(gold)),
            hit=hit, rank=found_rank, rr=rr,
        )

    n = len(rows)
    print(f"\n  hit_rate={hits/n:.4f}  MRR={mrr_sum/n:.4f}  n={n}")


# ── Result display ─────────────────────────────────────────────────────────────

def print_result(result: dict, elapsed_ms: int):
    print()
    sq = result.get("search_question", "")
    q = result.get("question", "")
    if sq and sq != q:
        print(f"[Rewritten query] {sq}")

    for i, (src, score) in enumerate(
        zip(result.get("sources", []), result.get("scores", [])), 1
    ):
        print(f"  Doc {i}: {src}  [score={score}]")

    pt = result.get("prompt_tokens")
    ct = result.get("completion_tokens")
    token_info = f"  tokens={pt}+{ct}" if pt and ct else ""

    print(f"\nAnswer:\n{result['answer']}")
    print(
        f"\n[{elapsed_ms} ms | context={result.get('context_chars', 0)} chars"
        f"{token_info} | lang={result.get('lang', '?')}]"
    )
    print()


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    cfg = Config.from_env()
    model, vectorstore, chunker, generator, retriever, metrics = build_app(cfg)

    print("=" * 60)
    print("  Telecom Multimodal RAG")
    print("=" * 60)
    print(f"  Model   : {cfg.chat_model}  |  Embed: {cfg.embed_model}")
    print(f"  DB      : {cfg.persist_dir}  |  Collection: {cfg.collection_name}")
    print(f"  Retrieval: k={cfg.retrieval_k}  score≥{cfg.score_threshold}  max_docs={cfg.max_context_docs}")
    print(f"  Tokens  : highlight={cfg.highlight_sentences} sentences  max_chunk={cfg.max_chunk_chars} chars")
    print(f"  History : compress after {cfg.max_history_turns} turns")
    print(f"  DB count: {vectorstore.count()}")
    print()
    print("  Commands: /ingest /count /where /clear_where /ask /history /export /eval  |  quit")
    print()

    default_where: dict = {}

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if not raw:
            continue

        cmd, _, rest = raw.partition(" ")
        cmd = cmd.lower()

        # ── Commands ───────────────────────────────────────────────────────────
        if cmd in ("quit", "exit"):
            print("Goodbye.")
            break

        elif cmd == "/ingest":
            path = rest.strip().strip('"')
            if path:
                ingest([path], chunker, vectorstore, model, metrics)
                print(f"  DB count: {vectorstore.count()}")
            else:
                print("  Usage: /ingest <path>")

        elif cmd == "/count":
            print(f"  DB count: {vectorstore.count()}")

        elif cmd == "/where":
            if rest.strip():
                default_where = parse_where(rest)
                print(f"  Filter set: {default_where}")
            else:
                print("  Usage: /where <key=val ...>")

        elif cmd == "/clear_where":
            default_where = {}
            print("  Filter cleared.")

        elif cmd == "/history":
            if rest.strip() == "clear":
                generator._history.clear()
                print("  History cleared.")
            else:
                print(f"  History turns: {generator._history.turn_count}")

        elif cmd == "/export":
            out = rest.strip() or cfg.metrics_dir
            metrics.export_csv(out)
            print(f"  Metrics exported to: {os.path.abspath(out)}")

        elif cmd == "/eval":
            eval_path = rest.strip().strip('"')
            if eval_path:
                run_eval(eval_path, retriever, metrics)
            else:
                print("  Usage: /eval <eval.jsonl>")

        elif cmd == "/ask":
            # /ask where:key=val key2=val2 <question>
            where = dict(default_where)
            question = rest.strip()
            if question.startswith("where:"):
                tail = question[6:]
                if " " in tail:
                    where_str, question = tail.split(" ", 1)
                    where.update(parse_where(where_str))
                else:
                    print("  Usage: /ask where:<key=val> <question>")
                    continue
            t = Timer()
            result = generator.ask(question.strip(), where=where or None)
            metrics.log_query(
                question=result["question"],
                search_question=result["search_question"],
                lang=result["lang"],
                sources=json.dumps(result["sources"]),
                scores=json.dumps(result["scores"]),
                context_chars=result["context_chars"],
                prompt_tokens=result.get("prompt_tokens"),
                completion_tokens=result.get("completion_tokens"),
                latency_ms=t.elapsed_ms(),
            )
            print_result(result, t.elapsed_ms())

        else:
            # Plain question — use default filter
            t = Timer()
            result = generator.ask(raw, where=default_where or None)
            metrics.log_query(
                question=result["question"],
                search_question=result["search_question"],
                lang=result["lang"],
                sources=json.dumps(result["sources"]),
                scores=json.dumps(result["scores"]),
                context_chars=result["context_chars"],
                prompt_tokens=result.get("prompt_tokens"),
                completion_tokens=result.get("completion_tokens"),
                latency_ms=t.elapsed_ms(),
            )
            print_result(result, t.elapsed_ms())


if __name__ == "__main__":
    main()
