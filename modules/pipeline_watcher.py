import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.pipeline_extractor import (
    get_pipeline_insights,
    list_staging_traces,
    _extract_question_from_trace,
)
from modules.pipeline_judge import judge_pipeline

PIPELINE_EVAL_PATH = Path(__file__).resolve().parents[1] / "output" / "evaluations_pipeline.jsonl"


# --- anti-duplikasi ---


def _load_evaluated_trace_ids():
    """Set trace_id yang sudah dievaluasi dari evaluations_pipeline.jsonl."""
    ids = set()
    if PIPELINE_EVAL_PATH.exists():
        with open(PIPELINE_EVAL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = rec.get("trace_id")
                if tid and rec.get("total") is not None and not rec.get("error"):
                    ids.add(tid)
    return ids


def _save_eval(rec: dict):
    PIPELINE_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_EVAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _build_rec(trace_id, question, result) -> dict:
    return {
        "type": "pipeline_eval",
        "timestamp": datetime.now().isoformat(),
        "trace_id": trace_id,
        "question": question,
        "intent_understanding": result.get("intent_understanding"),
        "query_expansion": result.get("query_expansion"),
        "reasoning": result.get("reasoning"),
        "memory_continuity": result.get("memory_continuity"),
        "total": result.get("total"),
        "kesimpulan": result.get("kesimpulan"),
    }


def evaluate_new_traces(hours: int = 24, limit: int = 200, on_progress=None):
    """Baca trace baru dari staging, nilai pipeline, simpan hasil.

    Langkah per trace: ekstrak pertanyaan -> insights -> judge -> simpan.
    Return dict ringkasan {'total', 'ok', 'gagal', 'detail'}."""
    evaluated = _load_evaluated_trace_ids()
    traces = list_staging_traces(hours=hours, limit=limit)

    new_traces = [t for t in traces if t["trace_id"] not in evaluated]
    detail = []
    ok = 0
    gagal = 0

    for i, t in enumerate(new_traces, start=1):
        trace_id = t["trace_id"]
        if on_progress:
            on_progress(f"Trace {i}/{len(new_traces)}: {trace_id[:12]}...")

        try:
            question = _extract_question_from_trace(trace_id)
            if not question:
                detail.append({"trace_id": trace_id, "status": "tanpa_pertanyaan"})
                gagal += 1
                continue
        except Exception as e:
            print(f"[WATCHER][{trace_id}] Gagal ekstrak pertanyaan: {e}")
            detail.append({"trace_id": trace_id, "status": "error", "error": str(e)})
            gagal += 1
            continue

        try:
            insights = get_pipeline_insights(None, None, trace_id=trace_id)
        except Exception as e:
            print(f"[WATCHER][{trace_id}] Gagal ambil insight: {e}")
            insights = None
        if not insights:
            detail.append({"trace_id": trace_id, "status": "tanpa_insight"})
            gagal += 1
            continue

        try:
            result = judge_pipeline(question, insights)
        except Exception as e:
            print(f"[WATCHER][{trace_id}] Gagal judge: {e}")
            detail.append({"trace_id": trace_id, "status": "error", "error": str(e)})
            gagal += 1
            continue
        if not result or "error" in result:
            detail.append({"trace_id": trace_id, "status": "judge_error"})
            gagal += 1
            continue

        _save_eval(_build_rec(trace_id, question, result))
        ok += 1
        detail.append(
            {
                "trace_id": trace_id,
                "status": "ok",
                "question": question,
                "total": result.get("total"),
                "insights": insights,
                "result": result,
            }
        )

    return {"total": len(new_traces), "ok": ok, "gagal": gagal, "detail": detail}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Watcher evaluasi pipeline dari trace staging.")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    summary = evaluate_new_traces(hours=args.hours, limit=args.limit)
    print(f"Trace baru: {summary['total']} | OK: {summary['ok']} | Gagal: {summary['gagal']}")
    for d in summary["detail"]:
        print(d)


if __name__ == "__main__":
    main()
