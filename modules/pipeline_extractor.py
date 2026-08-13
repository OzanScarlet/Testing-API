import json
import os
import re
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv
from phoenix.client import Client

load_dotenv()

CHATOPA_URL = os.getenv("CHATOPA_URL")
CHATOPA_API_KEY = os.getenv("CHATOPA_API_KEY")
PHOENIX_BASE_URL = os.getenv("PHOENIX_BASE_URL")
PHOENIX_RETRIEVE_PROJECT = os.getenv("PHOENIX_RETRIEVE_PROJECT") or os.getenv("PHOENIX_PROJECT_NAME")

POLL_INTERVAL = 2.0
POLL_TIMEOUT = 60.0

EXTRACTOR_MARKERS = ("retrieval_plan", "paraphrased_question")


def _first(container, key):
    """Ambil nilai defensif: dict biasa / dict dengan kunci `value` berlapis."""
    if not isinstance(container, dict):
        return None
    direct = container.get(key)
    if direct is not None:
        return direct
    inner = container.get("value")
    if isinstance(inner, dict):
        return inner.get(key)
    return None


def _deep_get(data, keys, default=None):
    """Ambil nilai dari dict bersarang; keys = list kunci bertingkat."""
    cur = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_request_id(response):
    return response.get("data", {}).get("request_id")


def _parse_llm_text(output_value):
    """output.value span ChatOpenAI umum: {'generations': [[{'text': ...}]]}.
    Kembalikan teks mentah LLM."""
    if not isinstance(output_value, str):
        return None
    try:
        data = json.loads(output_value)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not data.get("generations"):
        return None
    gen = data["generations"][0]
    if isinstance(gen, list) and gen:
        gen = gen[0]
    if not isinstance(gen, dict):
        return None
    return gen.get("text")


def _parse_extractor_json(output_value):
    """Parse output LLM span extractor menjadi dict JSON (defensif, skema beragam)."""
    if not isinstance(output_value, str):
        return {}
    text = _parse_llm_text(output_value)
    if not text:
        try:
            direct = json.loads(output_value)
            if isinstance(direct, dict):
                return direct
        except (ValueError, TypeError):
            pass
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _extractor_from_output(data):
    """Normalisasi output extractor dari skema lama (top-level) maupun
    skema baru (bungkus 'extractor' / 'query')."""
    out = {}
    if not isinstance(data, dict):
        return out

    wrapped = data.get("extractor") if isinstance(data.get("extractor"), dict) else data
    query = wrapped.get("query") if isinstance(wrapped.get("query"), dict) else wrapped
    plan = data.get("retrieval_plan") if isinstance(data.get("retrieval_plan"), dict) else wrapped

    out["intent"] = _first(wrapped, "intent")
    out["topics"] = _as_list(_first(wrapped, "topics"))
    out["is_topic_shift"] = _first(wrapped, "is_topic_shift")
    out["topic_action"] = _first(wrapped, "topic_action")
    out["paraphrased_question"] = _first(query, "paraphrased_question")
    out["primary_query"] = _first(query, "primary_query") or _first(plan, "primary_query")
    out["image_context"] = _first(query, "image_context")
    out["memory_queries"] = _as_list(_first(wrapped, "memory_queries"))
    out["query_expansions"] = _as_list(_first(plan, "query_expansions"))
    out["retrieval_step"] = _first(plan, "step")
    out["skip_retrieval"] = _first(plan, "skip_retrieval")
    out["retrieval_mode"] = _first(plan, "retrieval_mode")
    out["target_collections"] = _as_list(_first(plan, "target_collections"))
    out["target_documents"] = _as_list(_first(plan, "target_documents"))
    out["reasoning"] = _first(plan, "reasoning")
    amb = data.get("ambiguity")
    out["ambiguity"] = amb if isinstance(amb, dict) else None
    out["raw_extractor"] = wrapped
    out["raw_plan"] = plan
    return out


def _extractor_from_input(input_value):
    """Output extractor juga bisa tersimpan di bagian 'extractor' dari input
    span extractor_stage / preprocess_and_gate (turn berisi session dll).
    Kembalikan dict hasil parse, kosong bila tidak cocok."""
    out = {}
    if not isinstance(input_value, str):
        return out
    try:
        data = json.loads(input_value)
    except (ValueError, TypeError):
        return out
    if not isinstance(data, dict):
        return out

    turn = data.get("turn") if isinstance(data.get("turn"), dict) else {}
    out["turn_question"] = turn.get("question")
    out["history"] = _as_list(turn.get("history"))
    out["memory_window"] = _first(turn, "memory_window")
    session_state = data.get("session_state") or turn.get("session_state")
    out["session_state"] = session_state if isinstance(session_state, dict) else {}
    out["conversation_history"] = _as_list(data.get("conversation_history"))
    out["conversation_summary"] = data.get("conversation_summary")
    out["memory_context"] = data.get("memory_context") or turn.get("memory_context")
    out["context_window"] = data.get("context_window")

    for key in ("extractor", "gate"):
        sub = data.get(key)
        if isinstance(sub, dict):
            out.update(_extractor_from_output(sub))
            break
    return out


def _find_trace_id(request_id, question):
    client = Client(base_url=PHOENIX_BASE_URL)
    deadline = time.time() + POLL_TIMEOUT
    norm_question = " ".join(question.split())

    while time.time() < deadline:
        try:
            import pandas as pd

            df = client.spans.get_spans_dataframe(
                project_name=PHOENIX_RETRIEVE_PROJECT,
                start_time=datetime.now() - timedelta(minutes=3),
                limit=1000,
                timeout=30,
            )
        except Exception as e:
            print(f"[PHOENIX] Gagal query spans (timeout/error): {e}")
            time.sleep(POLL_INTERVAL)
            continue
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                metadata = row.get("attributes.metadata") or {}
                matches = (request_id and metadata.get("request_id") == request_id) or (
                    norm_question
                    and norm_question in " ".join(str(row.get("attributes.input.value", "")).split())
                )
                if matches:
                    trace_id = row.get("context.trace_id")
                    if trace_id:
                        return trace_id
        time.sleep(POLL_INTERVAL)

    return None


def list_staging_traces(hours: int = 24, limit: int = 200):
    """Daftarkan trace dari project retrieval (staging) tanpa POST.

    Return list dict: [{'trace_id', 'start_time'}], terurut menurun
    berdasarkan waktu mulai."""
    client = Client(base_url=PHOENIX_BASE_URL)
    try:
        traces = client.traces.get_traces(
            project_identifier=PHOENIX_RETRIEVE_PROJECT,
            start_time=datetime.now() - timedelta(hours=hours),
            sort="start_time",
            order="desc",
            include_spans=False,
            limit=limit,
            timeout=30,
        )
    except Exception as e:
        print(f"[PHOENIX] Gagal list trace staging: {e}")
        return []
    out = []
    for t in traces or []:
        trace_id = (t or {}).get("trace_id")
        if not trace_id:
            continue
        out.append({"trace_id": trace_id, "start_time": (t or {}).get("start_time")})
    return out


def _extract_question_from_trace(trace_id):
    """Ekstrak pertanyaan user dari input span extractor/preprocess pada trace.

    Return string pertanyaan atau None bila tidak ditemukan."""
    client = Client(base_url=PHOENIX_BASE_URL)
    try:
        spans = client.spans.get_spans(
            project_identifier=PHOENIX_RETRIEVE_PROJECT,
            trace_ids=[trace_id],
            limit=200,
            timeout=30,
        )
    except Exception as e:
        print(f"[PHOENIX] Gagal get_spans trace {trace_id}: {e}")
        return None
    for span in spans or []:
        attrs = span.get("attributes") or {}
        name = span.get("name") or ""
        if "extractor" not in name.lower() and "preprocess" not in name.lower():
            continue
        input_value = attrs.get("input.value")
        parsed = _extractor_from_input(input_value) if isinstance(input_value, str) else {}
        if parsed.get("turn_question"):
            return parsed["turn_question"]
    for span in spans or []:
        attrs = span.get("attributes") or {}
        if span.get("span_kind") == "LLM":
            output = attrs.get("output.value")
            ext = _extractor_from_output(_parse_extractor_json(output)) if isinstance(output, str) else {}
            if ext.get("paraphrased_question"):
                return ext["paraphrased_question"]
    return None

def get_pipeline_insights(request_id=None, question=None, trace_id=None):
    """Ambil insight pipeline dari trace Phoenix:
    output extractor (pemahaman, query expansion, reasoning) + input
    (histori turn / session_state / memory) untuk penilaian baru.

    `trace_id` opsional: langsung akses trace tanpa polling/request_id.
    Return dict terstruktur, atau None bila trace/insight tidak ditemukan."""
    if not trace_id:
        trace_id = _find_trace_id(request_id, question)
    if trace_id is None:
        return None

    client = Client(base_url=PHOENIX_BASE_URL)
    deadline = time.time() + POLL_TIMEOUT
    results = {}

    while time.time() < deadline:
        try:
            spans = client.spans.get_spans(
                project_identifier=PHOENIX_RETRIEVE_PROJECT,
                trace_ids=[trace_id],
                limit=200,
                timeout=30,
            )
        except Exception as e:
            print(f"[PHOENIX] Gagal get_spans (timeout/error): {e}")
            time.sleep(POLL_INTERVAL)
            continue

        for span in spans:
            attrs = span.get("attributes") or {}
            name = span.get("name") or ""
            span_kind = span.get("span_kind") or ""

            if "extractor" in name.lower() or "preprocess" in name.lower():
                for side in ("input", "output"):
                    value = attrs.get(f"{side}.value")
                    if not isinstance(value, str):
                        continue
                    parsed = _extractor_from_input(value)
                    if parsed.get("history") is not None or parsed.get("session_state"):
                        results.setdefault("input", {})
                        results["input"].setdefault(side, {})
                        results["input"][side].update(parsed)
                    ext = _extractor_from_output(_try_json(value))
                    if "intent" in ext:
                        results.setdefault("extractor", {})
                        results["extractor"].setdefault(side, {})
                        results["extractor"][side].update(ext)

            if span_kind == "LLM":
                output = attrs.get("output.value")
                if isinstance(output, str) and any(m in output for m in EXTRACTOR_MARKERS):
                    ext = _extractor_from_output(_parse_extractor_json(output))
                    if "intent" in ext or "paraphrased_question" in ext:
                        results.setdefault("extractor", {})
                        results["extractor"].setdefault("llm", {})
                        results["extractor"]["llm"].update(ext)

        if "extractor" in results and "input" in results:
            break
        if "extractor" in results:
            break
        time.sleep(POLL_INTERVAL)

    if "extractor" not in results and "input" not in results:
        return None

    return _merge_insights(results, question)


def _try_json(value):
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def _merge_insights(results, question):
    """Gabungkan semua data yang terkumpul menjadi satu dict insight terstruktur."""
    ext_llm = results.get("extractor", {}).get("llm", {}) or {}
    ext_in = results.get("extractor", {}).get("input", {}) or {}
    ext_out = results.get("extractor", {}).get("output", {}) or {}
    inp_in = results.get("input", {}).get("input", {}) or {}
    inp_out = results.get("input", {}).get("output", {}) or {}

    def pick(*sources):
        for s in sources:
            if s:
                return s
        return {}

    ex = pick(ext_llm, ext_out, ext_in)
    tx = pick(inp_out, inp_in)

    insights = {
        "question": tx.get("turn_question") or question,
        # --- pemahaman ---
        "intent": _first(ex, "intent"),
        "topics": _first(ex, "topics") or [],
        "is_topic_shift": _first(ex, "is_topic_shift"),
        "topic_action": _first(ex, "topic_action"),
        "paraphrased_question": _first(ex, "paraphrased_question"),
        "image_context": _first(ex, "image_context"),
        "ambiguity": ex.get("ambiguity"),
        # --- query expansion ---
        "primary_query": _first(ex, "primary_query"),
        "query_expansions": _first(ex, "query_expansions") or [],
        "retrieval_step": _first(ex, "retrieval_step"),
        "skip_retrieval": _first(ex, "skip_retrieval"),
        "retrieval_mode": _first(ex, "retrieval_mode"),
        "target_collections": _first(ex, "target_collections") or [],
        "target_documents": _first(ex, "target_documents") or [],
        # --- reasoning ---
        "reasoning": _first(ex, "reasoning"),
        # --- memori ---
        "memory_queries": _first(ex, "memory_queries") or tx.get("memory_queries") or [],
        "history": tx.get("history") or [],
        "memory_window": tx.get("memory_window"),
        "session_state": tx.get("session_state") or {},
        "conversation_history": tx.get("conversation_history") or [],
        "conversation_summary": tx.get("conversation_summary"),
        "memory_context": tx.get("memory_context"),
        "context_window": tx.get("context_window"),
    }
    return insights


def main():
    if not CHATOPA_URL or not CHATOPA_API_KEY:
        print("Konfigurasi CHATOPA_URL / CHATOPA_API_KEY belum lengkap di .env.")
        return
    if not PHOENIX_BASE_URL or not PHOENIX_RETRIEVE_PROJECT:
        print("Konfigurasi PHOENIX_BASE_URL / PHOENIX_RETRIEVE_PROJECT belum lengkap di .env.")
        return

    question = input("Pertanyaan: ").strip()
    if not question:
        print("Pertanyaan kosong.")
        return

    headers = {"x-api-key": CHATOPA_API_KEY}
    data = {"content": question}

    print(f"POST {CHATOPA_URL}")
    r = requests.post(CHATOPA_URL, headers=headers, data=data, timeout=180)
    r.raise_for_status()
    response = r.json()
    request_id = extract_request_id(response)
    print(f"Request id: {request_id}")

    print("Ambil insight pipeline dari trace Phoenix...")
    insights = get_pipeline_insights(request_id, question)
    if insights is None:
        print("Insight tidak ditemukan. Cek tracing/retrieval aktif.")
        return
    import pprint

    pprint.pprint(insights, width=120)


if __name__ == "__main__":
    main()