import json
import os
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


def extract_request_id(response):
    return response.get("data", {}).get("request_id")


def parse_answer(output_value):
    """output.value dari langchain berupa JSON {'generations': [[{'text': ...}]]}.
    Teksnya sendiri JSON string {'content': ..., 'citation_ids': ..., 'suggestions': ...},
    jawaban ada di 'content'."""
    if not isinstance(output_value, str):
        return str(output_value)
    try:
        data = json.loads(output_value)
    except (ValueError, TypeError):
        return output_value

    text = None
    if isinstance(data, dict) and data.get("generations"):
        gen = data["generations"][0]
        if isinstance(gen, list) and gen:
            gen = gen[0]
        if isinstance(gen, dict):
            text = gen.get("text")
    if text is None:
        return output_value

    try:
        inner = json.loads(text)
        if isinstance(inner, dict) and inner.get("content"):
            return inner["content"]
        return text
    except (ValueError, TypeError):
        return text


def _find_trace_id(request_id, question):
    """Polling cari LLM span yang request_id / teks pertanyaannya cocok,
    kembalikan context.trace_id (atau None)."""
    client = Client(base_url=PHOENIX_BASE_URL)
    deadline = time.time() + POLL_TIMEOUT
    norm_question = " ".join(question.split())

    while time.time() < deadline:
        try:
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
                if row.get("span_kind") != "LLM":
                    continue
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


def get_answer_from_trace(trace_id):
    """GET trace spesifik via trace_ids, ambil jawaban dari LLM span yang
    output-nya berisi kunci 'content' (jawaban final), bukan plan/intent."""
    client = Client(base_url=PHOENIX_BASE_URL)
    spans = client.spans.get_spans(
        project_identifier=PHOENIX_RETRIEVE_PROJECT,
        trace_ids=[trace_id],
        limit=100,
        timeout=30,
    )
    llm_spans = [s for s in spans if s.get("span_kind") == "LLM"]
    if not llm_spans:
        return None
    for span in llm_spans:
        attrs = span.get("attributes") or {}
        output = attrs.get("output.value")
        if output is None:
            continue
        parsed = parse_answer(output)
        stripped = parsed.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            continue
        if stripped:
            return parsed
    return None


def get_retrieval_context(request_id, question):
    """Cari trace via polling, lalu ambil konteks retrieval dari span
    retrieve_context_and_extract. Output.value bisa berupa JSON dict
    (retrieval.context) atau string repr Python (retrieval=RetrievalOutput(context="...")).
    Return string atau None."""
    trace_id = _find_trace_id(request_id, question)
    if trace_id is None:
        return None

    client = Client(base_url=PHOENIX_BASE_URL)
    deadline = time.time() + POLL_TIMEOUT

    while time.time() < deadline:
        try:
            spans = client.spans.get_spans(
                project_identifier=PHOENIX_RETRIEVE_PROJECT,
                trace_ids=[trace_id],
                limit=100,
                timeout=30,
            )
        except Exception as e:
            print(f"[PHOENIX] Gagal get_spans (timeout/error): {e}")
            time.sleep(POLL_INTERVAL)
            continue

        for span in spans:
            if span.get("name") != "retrieve_context_and_extract":
                continue
            attrs = span.get("attributes") or {}
            output_value = attrs.get("output.value")
            if not isinstance(output_value, str):
                continue
            context = _extract_retrieval_context(output_value)
            if context:
                return context

        time.sleep(POLL_INTERVAL)

    return None


def _extract_retrieval_context(output_value):
    """Ekstrak konteks retrieval dari output.value span retrieve_context_and_extract.
    Mendukung format JSON dict maupun string repr Python."""
    try:
        data = json.loads(output_value)
    except (ValueError, TypeError):
        return None

    if isinstance(data, dict):
        retrieval = data.get("retrieval") or {}
        context = retrieval.get("context")
        if isinstance(context, str) and context.strip():
            return context
        return None

    if isinstance(data, str):
        marker = "retrieval=RetrievalOutput(context="
        start = data.find(marker)
        if start == -1:
            return None
        quote = data.find('"', start + len(marker))
        if quote == -1:
            return None
        end = data.find('",', quote + 1)
        if end == -1:
            return None
        context = data[quote + 1 : end]
        if context.strip():
            return context

    return None


def find_answer_in_phoenix(request_id, question):
    """Cari trace_id via polling, lalu GET trace spesifik dan ambil jawabannya."""
    trace_id = _find_trace_id(request_id, question)
    if trace_id is None:
        return None
    return get_answer_from_trace(trace_id)


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
    try:
        r = requests.post(CHATOPA_URL, headers=headers, data=data, timeout=180)
        print(f"Status: {r.status_code}")
        if r.status_code != 200:
            print(f"Response: {r.text[:800]}")
            print("POST gagal, tidak dilanjutkan.")
            return
    except Exception as e:
        print(f"Gagal terhubung ke {CHATOPA_URL}: {e}")
        return

    response = r.json()
    request_id = extract_request_id(response)
    print(f"Request id: {request_id}")
    if not request_id:
        print("request_id tidak ditemukan di response. Cek struktur response.")
        return

    print(f"GET jawaban dari Phoenix: {PHOENIX_BASE_URL} (project={PHOENIX_RETRIEVE_PROJECT})")
    answer = find_answer_in_phoenix(request_id, question)
    if answer is not None:
        print(f"\nJawaban dari Phoenix ({request_id}):\n{answer[:1000]}")
    else:
        print(f"Trace untuk request {request_id} tidak ditemukan dalam {POLL_TIMEOUT:.0f} detik.")


if __name__ == "__main__":
    main()
