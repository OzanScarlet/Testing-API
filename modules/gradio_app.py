import os
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.judge import get_answer, judge_answer
from modules.phoenix_extractor import find_answer_in_phoenix

load_dotenv()


def run_eval(question: str) -> tuple:
    """Alur: POST ke chatbot -> ambil trace dari Phoenix -> judge 3 kriteria."""
    if not question or not question.strip():
        return "Pertanyaan kosong.", "", ""

    try:
        answer, request_id = get_answer(question)
    except Exception as e:
        return f"Gagal POST ke chatbot: {e}", "", ""
    if not answer:
        return "Jawaban tidak ditemukan di response POST.", "", ""

    fastapi_text = f"Jawaban (FastAPI):\n{answer}\n\nRequest id: {request_id}"

    try:
        trace_answer = find_answer_in_phoenix(request_id, question)
    except Exception as e:
        trace_answer = None

    if trace_answer:
        trace_text = f"Jawaban (trace):\n{trace_answer}"
    else:
        trace_text = "Trace tidak ditemukan di Phoenix (tracing staging mungkin mati)."

    if trace_answer:
        try:
            result = judge_answer(question, answer, trace_answer)
        except Exception as e:
            result = {"error": str(e)}
    else:
        result = None

    if result is None:
        judge_text = "Judge tidak dijalankan karena tidak ada acuan trace."
    elif "error" in result:
        judge_text = f"Judge gagal: {result['error']}"
    else:
        lines = []
        for key in ("akurasi", "kelengkapan", "kesesuaian"):
            item = result.get(key, {})
            lines.append(f"- {key}: {item.get('skor')}/10 — {item.get('alasan')}")
        lines.append(f"- Total: {result.get('total')}/10 ({result.get('label')})")
        lines.append(f"- Kesimpulan: {result.get('kesimpulan')}")
        judge_text = "\n".join(lines)

    return fastapi_text, trace_text, judge_text


def main():
    if not os.getenv("CHATOPA_URL") or not os.getenv("CHATOPA_API_KEY"):
        print("Konfigurasi CHATOPA_URL / CHATOPA_API_KEY belum lengkap di .env.")
    if not os.getenv("BASE_URL") or not os.getenv("OPENAI_API_KEY"):
        print("Konfigurasi BASE_URL / OPENAI_API_KEY belum lengkap di .env.")

    with gr.Blocks(title="Evaluasi Chatbot") as demo:
        gr.Markdown("# Evaluasi Chatbot RAG Kelapa Sawit")
       

        question = gr.Textbox(
            label="Pertanyaan",
            placeholder="Contoh: Apa perbedaan sawit dan kelapa?",
            lines=2,
        )
        btn = gr.Button("Evaluasi", variant="primary")

        fastapi_out = gr.Textbox(label=" FastAPI", lines=6, interactive=False)
        trace_out = gr.Textbox(label=" Trace Phoenix", lines=6, interactive=False)
        judge_out = gr.Textbox(label="Hasil Judge", lines=8, interactive=False)

        btn.click(run_eval, inputs=[question], outputs=[fastapi_out, trace_out, judge_out])

    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
