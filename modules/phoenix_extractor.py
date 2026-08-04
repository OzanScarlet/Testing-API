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
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME")

POLL_INTERVAL = 2.0
POLL_TIMEOUT = 30.0


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


def find_answer_in_phoenix(request_id, question):
    """Cari LLM span di Phoenix yang request_id atau teks pertanyaannya cocok,
    lalu ambil output jawabannya. Polling sampai trace muncul."""
    client = Client(base_url=PHOENIX_BASE_URL)
    deadline = time.time() + POLL_TIMEOUT
    norm_question = " ".join(question.split())

    while time.time() < deadline:
        df = client.spans.get_spans_dataframe(
            project_name=PHOENIX_PROJECT_NAME,
            start_time=datetime.now() - timedelta(minutes=3),
            limit=1000,
        )
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
                    output = row.get("attributes.output.value")
                    if output is not None:
                        return parse_answer(output)
        time.sleep(POLL_INTERVAL)

    return None


def main():
    if not CHATOPA_URL or not CHATOPA_API_KEY:
        print("Konfigurasi CHATOPA_URL / CHATOPA_API_KEY belum lengkap di .env.")
        return
    if not PHOENIX_BASE_URL or not PHOENIX_PROJECT_NAME:
        print("Konfigurasi PHOENIX_BASE_URL / PHOENIX_PROJECT_NAME belum lengkap di .env.")
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

    print(f"GET jawaban dari Phoenix: {PHOENIX_BASE_URL} (project={PHOENIX_PROJECT_NAME})")
    answer = find_answer_in_phoenix(request_id, question)
    if answer is not None:
        print(f"\nJawaban dari Phoenix ({request_id}):\n{answer[:1000]}")
    else:
        print(f"Trace untuk request {request_id} tidak ditemukan dalam {POLL_TIMEOUT:.0f} detik.")


if __name__ == "__main__":
    main()
