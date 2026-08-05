import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.phoenix_extractor import find_answer_in_phoenix

load_dotenv()

CHATOPA_URL = os.getenv("CHATOPA_URL")
CHATOPA_API_KEY = os.getenv("CHATOPA_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
BASE_URL = os.getenv("BASE_URL")

MAX_RETRIES = 6
MAX_DELAY = 60.0
DELAY = float(os.getenv("DELAY", "3.0"))


def bump_delay():
    """Naikkan jeda secara adaptif."""
    global DELAY
    DELAY = min(DELAY * 2, MAX_DELAY)
    print(f"[RATE LIMIT] Delay dinaikkan menjadi {DELAY:.0f} detik.")


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=BASE_URL,
        default_headers={"User-Agent": "python-requests/2.32"},
    )


def get_answer(question: str) -> tuple:
    """POST pertanyaan ke chatbot FastAPI, kembalikan (jawaban, request_id)."""
    headers = {"x-api-key": CHATOPA_API_KEY}
    data = {"content": question}
    r = requests.post(CHATOPA_URL, headers=headers, data=data, timeout=180)
    r.raise_for_status()
    payload = r.json()
    answer = payload.get("data", {}).get("message", {}).get("content")
    request_id = payload.get("data", {}).get("request_id")
    return answer, request_id


def judge_answer(question: str, fastapi_answer: str, trace_answer: str) -> dict:
    """Nilai jawaban FastAPI dengan 3 kriteria, memakai jawaban trace sebagai acuan."""
    system_prompt = (
        "Kamu adalah evaluator kualitas jawaban chatbot RAG kelapa sawit. "
        "Nilai jawaban yang ditampilkan ke user (dari FastAPI) dengan acuan jawaban "
        "yang tercatat di trace (dari Phoenix) — trace dianggap sebagai jawaban yang "
        "dibuat sistem berdasarkan konteks dokumen.\n"
        "Nilai berdasarkan tiga kriteria:\n"
        "1. Akurasi pertanyaan dan jawaban: seberapa tepat dan benar jawaban menjawab "
        "pertanyaan dibanding acuan.\n"
        "2. Kelengkapan jawaban: seberapa lengkap jawaban mencakup poin penting yang ada "
        "di acuan.\n"
        "3. Kesesuaian jawaban dengan konteks: seberapa sesuai jawaban dengan isi acuan, "
        "tidak bertentangan.\n"
        "Berikan skor 0-10 per kriteria dengan alasan singkat, lalu total (rata-rata) "
        "dan kesimpulan.\n"
        "OUTPUT WAJIB BERFORMAT JSON DENGAN STRUKTUR KETAT TANPA KUNCI LAIN.\n"
        'Contoh:\n'
        '{\n'
        '  "akurasi": {"skor": 8, "alasan": "..."},\n'
        '  "kelengkapan": {"skor": 7, "alasan": "..."},\n'
        '  "kesesuaian": {"skor": 9, "alasan": "..."},\n'
        '  "total": 8.0,\n'
        '  "label": "Baik",\n'
        '  "kesimpulan": "..."\n'
        '}'
    )
    user_prompt = (
        f"Pertanyaan:\n{question}\n\n"
        f"Jawaban yang ditampilkan ke user (FastAPI):\n{fastapi_answer}\n\n"
        f"Acuan (jawaban di trace Phoenix):\n{trace_answer or '(tidak ada acuan)'}"
    )

    client = get_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(DELAY)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            data = json.loads(response.choices[0].message.content)
            for key in ("akurasi", "kelengkapan", "kesesuaian", "total", "label", "kesimpulan"):
                if key not in data:
                    raise ValueError(f"Kunci '{key}' hilang di output judge")
            return data
        except RateLimitError as e:
            bump_delay()
            wait = max(2 ** attempt, DELAY)
            print(f"[RATE LIMIT] Percobaan {attempt}/{MAX_RETRIES}. Tunggu {wait:.0f} detik...")
            time.sleep(wait)
        except Exception as e:
            print(f"[ERROR] Judge gagal (percobaan {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 ** attempt)

    raise RuntimeError("Judge gagal setelah beberapa percobaan.")


def main():
    if not CHATOPA_URL or not CHATOPA_API_KEY:
        print("Konfigurasi CHATOPA_URL / CHATOPA_API_KEY belum lengkap di .env.")
        return
    if not BASE_URL or not os.getenv("OPENAI_API_KEY"):
        print("Konfigurasi BASE_URL / OPENAI_API_KEY belum lengkap di .env.")
        return

    question = input("Pertanyaan: ").strip()
    if not question:
        print("Pertanyaan kosong.")
        return

    print(f"POST {CHATOPA_URL}")
    answer, request_id = get_answer(question)
    if not answer:
        print("Jawaban tidak ditemukan di response POST.")
        return
    print(f"Jawaban (FastAPI):\n{answer[:500]}")
    print(f"Request id: {request_id}")

    print("Ambil jawaban dari trace Phoenix...")
    trace_answer = find_answer_in_phoenix(request_id, question)
    if trace_answer is None:
        print("Trace untuk request ini tidak ditemukan di Phoenix. Coba lagi nanti.")
        return
    print(f"Jawaban (trace):\n{trace_answer[:500]}")

    print("Menilai jawaban...")
    result = judge_answer(question, answer, trace_answer)
    print("\n=== HASIL PENILAIAN ===")
    for key in ("akurasi", "kelengkapan", "kesesuaian"):
        item = result.get(key, {})
        print(f"- {key}: {item.get('skor')}/10 — {item.get('alasan')}")
    print(f"- Total: {result.get('total')}/10 ({result.get('label')})")
    print(f"- Kesimpulan: {result.get('kesimpulan')}")

if __name__ == "__main__":
    main()
