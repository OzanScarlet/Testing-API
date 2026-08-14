import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
BASE_URL = os.getenv("BASE_URL")

PHOENIX_BASE_URL = os.getenv("PHOENIX_BASE_URL")
PHOENIX_SEND_PROJECT = os.getenv("PHOENIX_SEND_PROJECT", "AutoAssesment")
PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT",
    f"{PHOENIX_BASE_URL.rstrip('/')}/v1/traces" if PHOENIX_BASE_URL else None,
)

MAX_RETRIES = 6
MAX_DELAY = 60.0
DELAY = float(os.getenv("DELAY", "3.0"))

_tracing_ready = False


def _ensure_tracing():
    """Pasang tracer evaluator -> PHOENIX_SEND_PROJECT (lazy, sekali)."""
    global _tracing_ready
    if _tracing_ready:
        return
    from modules.tracing_control import is_disabled

    if is_disabled():
        _tracing_ready = True
        return
    if not PHOENIX_COLLECTOR_ENDPOINT:
        print("[TRACE] PHOENIX_BASE_URL tidak diset — tracing evaluator dinonaktifkan.")
        _tracing_ready = True
        return
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from phoenix.otel import register

        OpenAIInstrumentor().instrument()
        register(
            project_name=PHOENIX_SEND_PROJECT,
            endpoint=PHOENIX_COLLECTOR_ENDPOINT,
            batch=True,
            protocol="http/protobuf",
            auto_instrument=False,
        )
        _tracing_ready = True
        print(
            f"[TRACE] Evaluator terinstrumentasi -> {PHOENIX_COLLECTOR_ENDPOINT} "
            f"(project {PHOENIX_SEND_PROJECT})."
        )
    except Exception as e:
        print(f"[TRACE] Gagal mengaktifkan tracing evaluator: {e}")
        _tracing_ready = True


def reset_tracing():
    """Buka kunci lazy init agar tracing bisa dipasang ulang."""
    global _tracing_ready
    _tracing_ready = False


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


def _insights_prompt(insights: dict) -> str:
    """Serialize insight pipeline menjadi teks prompt yang rapi."""
    def fmt(label, value):
        if value is None or value == [] or value == {}:
            return ""
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        return f"{label}: {value}\n"

    parts = []
    parts.append(f"Pertanyaan asli user: {insights.get('question')}")
    if insights.get("paraphrased_question"):
        parts.append(fmt("Pertanyaan yang diparafrase sistem", insights["paraphrased_question"]))
    if insights.get("primary_query"):
        parts.append(fmt("Primary query retrieval", insights["primary_query"]))
    if insights.get("query_expansions"):
        parts.append(fmt("Query expansions", insights["query_expansions"]))
    if insights.get("intent"):
        parts.append(fmt("Intent yang dideteksi", insights["intent"]))
    if insights.get("is_topic_shift") is not None:
        parts.append(fmt("is_topic_shift", insights["is_topic_shift"]))
    if insights.get("topic_action"):
        parts.append(fmt("topic_action", insights["topic_action"]))
    if insights.get("retrieval_step"):
        parts.append(fmt("Langkah retrieval (step)", insights["retrieval_step"]))
    if insights.get("skip_retrieval") is not None:
        parts.append(fmt("skip_retrieval", insights["skip_retrieval"]))
    if insights.get("retrieval_mode"):
        parts.append(fmt("Mode retrieval", insights["retrieval_mode"]))
    if insights.get("target_collections"):
        parts.append(fmt("Koleksi target", insights["target_collections"]))
    if insights.get("reasoning"):
        parts.append(fmt("Reasoning/alasan sistem", insights["reasoning"]))
    if insights.get("memory_queries"):
        parts.append(fmt("Memory queries (pencarian memori)", insights["memory_queries"]))
    if insights.get("history"):
        parts.append(fmt("Histori percakapan (turn.history)", insights["history"]))
    if insights.get("memory_window"):
        parts.append(fmt("Memory window", insights["memory_window"]))
    if insights.get("session_state"):
        parts.append(fmt("Session state", insights["session_state"]))
    if insights.get("conversation_summary"):
        parts.append(fmt("Ringkasan percakapan", insights["conversation_summary"]))
    if insights.get("memory_context"):
        parts.append(fmt("Memory context", insights["memory_context"]))
    if insights.get("ambiguity"):
        parts.append(fmt("Deteksi ambiguitas", insights["ambiguity"]))
    return "\n".join(p for p in parts if p)


def judge_pipeline(question: str, insights: dict, domain: str = "kelapa sawit") -> dict:
    """Nilai kualitas PIPELINE chatbot RAG (LLM Judge V2, skala 1-10)."""
    _ensure_tracing()

    system_prompt = (
        "Kamu adalah evaluator kualitas PIPELINE chatbot RAG. "
        "Kamu menganalisis JSON trace RAG yang dihasilkan sistem dan menilai "
        "PROSES: bagaimana sistem memahami pertanyaan, memperluas query, "
        "bernalar merencanakan retrieval, dan memanfaatkan memori percakapan.\n"
        f"Topik/tema domain: {domain}.\n"
        "Nilai berdasarkan empat kriteria (skala 1-10):\n"
        "1. Intent & Understanding: seberapa akurat `primary_query` dan `intent` "
        "hasil extractor mencerminkan maksud pertanyaan user. Nilai tinggi bila "
        "query utama setia pada maksud asli dan intent benar.\n"
        "2. Query Expansion: seberapa relevan variasi `query_expansions` "
        "(sinonim/istilah/parafrase) untuk memperkaya retrieval tanpa menyimpang "
        "dari maksud. PENTING: jika percakapan adalah casual chat atau TANPA "
        "retrieval (skip_retrieval=true / tidak ada data retrieval), otomatis "
        "nilai 10 karena perluasan query tidak diperlukan.\n"
        "3. Reasoning: seberapa logis dan akurat penalaran sistem saat "
        "merencanakan RAG pada field `reasoning` — pemilihan langkah/step, mode "
        "retrieval, koleksi target sesuai jenis pertanyaan.\n"
        "4. Memory Continuity: seberapa harmonis pemanfaatan memori percakapan — "
        "apakah `turn.history`, `session_state`, dan `extractor.memory_queries` "
        "selaras dengan konteks/perpindahan topik pertanyaan.\n"
        "Berikan skor 1-10 per kriteria dengan alasan singkat, lalu total "
        "(rata-rata) dan kesimpulan.\n"
        "OUTPUT WAJIB BERFORMAT JSON DENGAN STRUKTUR KETAT TANPA KUNCI LAIN.\n"
        'Contoh:\n'
        '{\n'
        '  "intent_understanding": {"skor": 8, "alasan": "..."},\n'
        '  "query_expansion": {"skor": 9, "alasan": "..."},\n'
        '  "reasoning": {"skor": 8, "alasan": "..."},\n'
        '  "memory_continuity": {"skor": 7, "alasan": "..."},\n'
        '  "total": 8.0,\n'
        '  "kesimpulan": "..."\n'
        '}'
    )
    user_prompt = (
        f"Data pipeline chatbot RAG:\n\n"
        f"{_insights_prompt(insights)}\n\n"
        f"Evaluasilah kualitas pipeline di atas berdasarkan 4 kriteria (skala 1-10)."
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
            for key in (
                "intent_understanding",
                "query_expansion",
                "reasoning",
                "memory_continuity",
                "total",
                "kesimpulan",
            ):
                if key not in data:
                    raise ValueError(f"Kunci '{key}' hilang di output judge")
            return data
        except RateLimitError as e:
            bump_delay()
            wait = max(2 ** attempt, DELAY)
            print(f"[RATE LIMIT] Percobaan {attempt}/{MAX_RETRIES}. Tunggu {wait:.0f} detik...")
            time.sleep(wait)
        except Exception as e:
            print(f"[ERROR] Judge pipeline gagal (percobaan {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 ** attempt)

    raise RuntimeError("Judge pipeline gagal setelah beberapa percobaan.")


def main():
    import pprint

    from modules.pipeline_extractor import get_pipeline_insights
    import requests

    if not BASE_URL or not os.getenv("OPENAI_API_KEY"):
        print("Konfigurasi BASE_URL / OPENAI_API_KEY belum lengkap di .env.")
        return

    question = input("Pertanyaan: ").strip()
    if not question:
        print("Pertanyaan kosong.")
        return

    from modules.pipeline_extractor import CHATOPA_URL, CHATOPA_API_KEY, extract_request_id

    headers = {"x-api-key": CHATOPA_API_KEY}
    r = requests.post(CHATOPA_URL, headers=headers, data={"content": question}, timeout=180)
    r.raise_for_status()
    request_id = extract_request_id(r.json())
    print(f"Request id: {request_id}")

    print("Ambil insight pipeline...")
    insights = get_pipeline_insights(request_id, question)
    if insights is None:
        print("Insight tidak ditemukan. Cek tracing aktif.")
        return
    print("Menilai pipeline...")
    result = judge_pipeline(question, insights)
    print("\n=== HASIL PENILAIAN PIPELINE ===")
    for key, label in (
        ("intent_understanding", "Intent & Understanding"),
        ("query_expansion", "Query Expansion"),
        ("reasoning", "Reasoning"),
        ("memory_continuity", "Memory Continuity"),
    ):
        item = result.get(key, {})
        print(f"- {label}: {item.get('skor')}/10 — {item.get('alasan')}")
    print(f"- Total: {result.get('total')}/10")
    print(f"- Kesimpulan: {result.get('kesimpulan')}")


if __name__ == "__main__":
    main()