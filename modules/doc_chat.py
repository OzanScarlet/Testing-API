import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.questions_generator import extract_text, split_document

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = ROOT / "output" / "qdrant_db"
COLLECTION = "docs"
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.3"))

MODEL_NAME = os.getenv("MODEL_NAME")
BASE_URL = os.getenv("BASE_URL")

PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix.rpn.my.id/v1/traces")
PHOENIX_PROJECT = os.getenv("DOC_CHAT_PHOENIX_PROJECT", "ChatOPA-staging")

MAX_RETRIES = 6
MAX_DELAY = 60.0
DELAY = float(os.getenv("DELAY", "3.0"))

_client = None
_client_qdrant = None
_embedder = None
_tracing_ready = False


def _init_tracing():
    """Pasang tracer Phoenix sekali (lazy) supaya panggilan OpenAI terekam."""
    global _tracing_ready
    if _tracing_ready:
        return
    try:
        from phoenix.otel import register

        register(
            project_name=PHOENIX_PROJECT,
            auto_instrument=True,
            batch=True,
            endpoint=PHOENIX_ENDPOINT,
            protocol="http/protobuf",
        )
        _tracing_ready = True
        print(f"[TRACE] Tracing aktif -> {PHOENIX_ENDPOINT} (project {PHOENIX_PROJECT}).")
    except Exception as e:
        print(f"[TRACE] Gagal mengaktifkan tracing: {e}")


def bump_delay():
    global DELAY
    DELAY = min(DELAY * 2, MAX_DELAY)
    print(f"[RATE LIMIT] Delay dinaikkan menjadi {DELAY:.0f} detik.")


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=BASE_URL,
            default_headers={"User-Agent": "python-requests/2.32"},
        )
    return _client


def _get_qdrant() -> QdrantClient:
    global _client_qdrant
    if _client_qdrant is None:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        _client_qdrant = QdrantClient(path=str(QDRANT_PATH))
    return _client_qdrant


def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource

    if EMBED_MODEL == "intfloat/multilingual-e5-small":
        from fastembed.common.model_description import PoolingType

        TextEmbedding.add_custom_model(
            EMBED_MODEL,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=EMBED_MODEL),
            dim=384,
            model_file="onnx/model.onnx",
            description="Multilingual E5 small",
            license="MIT",
            size_in_gb=0.1,
        )
    _embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=str(ROOT / "output" / ".fastembed"))
    return _embedder


def _embed(texts):
    return list(_get_embedder().embed(texts))


def index_documents(paths) -> int:
    """Hapus index lama lalu indeks dokumen baru. Return jumlah chunk."""
    qdrant = _get_qdrant()
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qdrant_models.VectorParams(size=384, distance=qdrant_models.Distance.COSINE),
    )

    points = []
    idx = 0
    for path in paths:
        text = extract_text(path)
        if not text.strip():
            print(f"[DOC] Lewati {Path(path).name}: tidak ada teks.")
            continue
        source = Path(path).name
        for chunk in split_document(text):
            if not chunk.strip():
                continue
            points.append((idx, chunk, source))
            idx += 1

    if not points:
        print("[DOC] Tidak ada chunk untuk diindeks.")
        return 0

    for i in range(0, len(points), 64):
        batch = points[i:i + 64]
        vectors = _embed([f"passage: {p[1]}" for p in batch])
        qdrant.upsert(
            collection_name=COLLECTION,
            points=[
                qdrant_models.PointStruct(
                    id=p[0],
                    vector=v.tolist(),
                    payload={"text": p[1], "source": p[2], "chunk_idx": p[0]},
                )
                for p, v in zip(batch, vectors)
            ],
        )

    print(f"[DOC] Diindeks {idx} chunk dari {len(paths)} file.")
    return idx


def _retrieve(query: str, top_k: int = None):
    top_k = top_k or RAG_TOP_K
    vector = _embed([f"query: {query}"])[0]
    qdrant = _get_qdrant()
    if not qdrant.collection_exists(COLLECTION):
        return []
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector.tolist(),
        limit=top_k,
        with_payload=True,
    ).points
    result = []
    for h in hits:
        result.append(
            {
                "text": (h.payload or {}).get("text", ""),
                "source": (h.payload or {}).get("source", "?"),
                "score": round(float(h.score), 4),
            }
        )
    return result


def _chat(messages: list, temperature: float = 0.3) -> str:
    client = get_client()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(DELAY)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            bump_delay()
            wait = max(2 ** attempt, DELAY)
            print(f"[RATE LIMIT] Percobaan {attempt}/{MAX_RETRIES}. Tunggu {wait:.0f} detik...")
            time.sleep(wait)
        except Exception as e:
            print(f"[ERROR] Chat dokumen gagal (percobaan {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError("Chat dokumen gagal setelah beberapa percobaan.")


def ask(question: str, history: list = None) -> tuple:
    """Tanya seputar dokumen. Return (jawaban, [sumber]).

    history: list of [user, assistant] pairs (bentuk Gradio Chatbot).
    Hybrid: konteks dokumen (top-k) selalu diberikan; LLM menjawab dari konteks
    dengan sitasi [n], dan jika pertanyaan di luar dokumen, menjawab dari
    pengetahuan umum dengan menyebut bahwa info tidak ada di dokumen.
    """
    history = history or []
    _init_tracing()
    hits = _retrieve(question)
    if hits:
        context = "\n\n".join(
            f"[{i + 1}] Sumber: {h['source']} (skor {h['score']})\n{h['text']}"
            for i, h in enumerate(hits)
        )
        system = (
            "Kamu adalah asisten RAG. Jawab pertanyaan berdasarkan konteks "
            "dokumen yang diberikan, dengan rujukan [n] sesuai nomor sumber.\n"
            "Jika pertanyaan TIDAK berkaitan dengan isi dokumen (konteks tidak "
            "membantu menjawab), tetap JAWAB PERTANYAANNYA dari pengetahuan umum "
            "secara lengkap dan singkat, lalu tambahkan catatan singkat bahwa "
            "informasi tersebut tidak tersedia di dokumen."
        )
        user = f"Konteks dokumen:\n{context}\n\nPertanyaan: {question}"
    else:
        system = (
            "Kamu adalah asisten yang membantu. Tidak ada dokumen yang diindeks, "
            "jadi jawab dari pengetahuan umum secara singkat."
        )
        user = question

    messages = [{"role": "system", "content": system}]
    for h_user, h_assistant in history[-6:]:
        messages.append({"role": "user", "content": h_user})
        messages.append({"role": "assistant", "content": h_assistant})
    messages.append({"role": "user", "content": user})

    answer = _chat(messages)
    return answer, hits


def main():
    print("Chat Dokumen (RAG). Ketik 'exit' untuk keluar.")
    history = []
    while True:
        question = input("\nPertanyaan: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break
        answer, hits = ask(question, history)
        print(f"\nJawaban: {answer}")
        if hits:
            print("\nSumber:")
            for h in hits:
                print(f"- {h['source']} (skor {h['score']})")
        history.append([question, answer])


if __name__ == "__main__":
    main()
