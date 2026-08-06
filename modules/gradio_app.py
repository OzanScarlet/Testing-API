import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.judge import get_answer, judge_answer
from modules.phoenix_extractor import get_retrieval_context

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "output" / "questions.json"
EVAL_PATH = ROOT / "output" / "evaluations.jsonl"

FILES = []
DONE = set()


def load_files():
    global FILES
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    FILES = data["files"]


def load_done():
    global DONE
    DONE = set()
    if not EVAL_PATH.exists():
        return
    with open(EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("question") and rec.get("total") is not None and not rec.get("error"):
                DONE.add(rec["question"])


def save_eval(rec: dict):
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_recap() -> pd.DataFrame:
    """Baca output/evaluations.jsonl -> DataFrame untuk tab rekapitulasi."""
    rows = []
    if EVAL_PATH.exists():
        with open(EVAL_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(
                    {
                        "Timestamp": rec.get("timestamp"),
                        "Pertanyaan": rec.get("question"),
                        "Total": rec.get("total"),
                        "Label": rec.get("label"),
                        "Request ID": rec.get("request_id"),
                    }
                )
    return pd.DataFrame(rows)


def _file_label(i: int) -> str:
    return f"{i + 1}. {FILES[i]['source']} ({len(FILES[i]['questions'])} soal)"


def _file_index(file_label) -> int:
    if not file_label:
        return None
    return int(str(file_label).split(".")[0]) - 1


def _next_unevaluated_index(questions, start):
    i = start
    while i < len(questions) and questions[i] in DONE:
        i += 1
    return i


def on_file_change(file_label, state: dict) -> tuple:
    """Pilih dokumen -> muat pertanyaan pertama yang belum dievaluasi."""
    state = dict(state) if state else {}
    idx = _file_index(file_label)
    state["file_label"] = file_label
    state["questions"] = FILES[idx]["questions"] if idx is not None else []
    qs = state["questions"]
    ni = _next_unevaluated_index(qs, 0)
    state["q_index"] = ni
    if ni < len(qs):
        state["question"] = qs[ni]
        return qs[ni], state
    state["question"] = ""
    return "✔ Semua pertanyaan pada dokumen ini sudah dievaluasi.", state


def advance_sequence(state: dict) -> tuple:
    """Setelah evaluasi: jika soal berhasil dinilai, maju ke soal berikutnya
    yang belum dievaluasi. Jika gagal (mis. konteks retrieval tidak ada),
    soal tetap dipertahankan agar bisa dicoba lagi (tidak dianggap selesai)."""
    state = dict(state) if state else {}
    qs = state.get("questions", [])
    current = state.get("question", "")
    if current and current not in DONE:
        return current, state
    ni = _next_unevaluated_index(qs, state.get("q_index", 0) + 1)
    state["q_index"] = ni
    if ni < len(qs):
        state["question"] = qs[ni]
        return qs[ni], state
    state["question"] = ""
    return "✔ Semua pertanyaan pada dokumen ini sudah dievaluasi.", state


def clear_all():
    empty_df = pd.DataFrame(columns=["Kriteria", "Skor", "Alasan"])
    return "", "", empty_df, None, ""


def _judge_dataframe(result) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["Kriteria", "Skor", "Alasan"])
    if not result or "error" in result:
        return empty
    rows = []
    for key, label in (
        ("akurasi", "Akurasi"),
        ("kelengkapan", "Kelengkapan"),
        ("kesesuaian", "Kesesuaian"),
    ):
        item = result.get(key, {})
        rows.append([label, item.get("skor"), item.get("alasan")])
    rows.append(["Total", result.get("total"), result.get("label")])
    return pd.DataFrame(rows, columns=["Kriteria", "Skor", "Alasan"])


def _judge_plot_df(result):
    if not result or "error" in result:
        return None
    rows = []
    for key, label in (
        ("akurasi", "Akurasi"),
        ("kelengkapan", "Kelengkapan"),
        ("kesesuaian", "Kesesuaian"),
    ):
        rows.append({"Kriteria": label, "Skor": result.get(key, {}).get("skor")})
    return pd.DataFrame(rows)


def _judge_summary(result) -> str:
    if result is None:
        return "Judge tidak dijalankan karena tidak ada konteks retrieval."
    if "error" in result:
        return f"**Judge gagal:** {result['error']}"
    return (
        f"**Total: {result.get('total')}/10 ({result.get('label')})**\n\n"
        f"{result.get('kesimpulan')}"
    )


MAX_EVAL_ATTEMPTS = 3
RETRY_DELAY = 5


def _empty_judge_df():
    return pd.DataFrame(columns=["Kriteria", "Skor", "Alasan"])


def _try_full_eval(question: str):
    """Satu percobaan lengkap: POST -> trace -> judge.
    Return dict sukses {answer, request_id, retrieval_context, result}
    atau dict {'error': ...}."""
    try:
        answer, request_id = get_answer(question)
    except Exception as e:
        print(f"[POST] error: {e}")
        return {"error": f"POST ke chatbot gagal: {e}"}
    if not answer:
        return {"error": "POST tidak mengembalikan jawaban."}

    try:
        retrieval_context = get_retrieval_context(request_id, question)
    except Exception as e:
        print(f"[TRACE] error: {e}")
        retrieval_context = None
    if not retrieval_context:
        return {"error": "Konteks retrieval tidak ditemukan."}

    try:
        result = judge_answer(question, answer, retrieval_context)
    except Exception as e:
        print(f"[JUDGE] error: {e}")
        return {"error": f"Judge gagal: {e}"}
    if not result or "error" in result:
        return result or {"error": "Judge tidak mengembalikan hasil."}

    return {
        "answer": answer,
        "request_id": request_id,
        "retrieval_context": retrieval_context,
        "result": result,
    }


def run_eval(manual: str, state: dict) -> tuple:
    """Evaluasi penuh dengan retry otomatis: POST -> trace -> judge diulang
    sampai semua step keluar hasil. Hanya soal yang BERHASIL semua step yang
    disimpan, ditandai selesai, dan dilanjutkan ke soal berikutnya."""
    question = (manual or "").strip() or (state or {}).get("question", "").strip()
    if not question:
        base = {"question": ""}
        return "Pertanyaan kosong.", "", _empty_judge_df(), None, "", "", base

    out = None
    last_err = None
    for attempt in range(1, MAX_EVAL_ATTEMPTS + 1):
        print(f"[EVAL] percobaan {attempt}/{MAX_EVAL_ATTEMPTS}: {question[:60]}")
        out = _try_full_eval(question)
        if out and "error" not in out:
            break
        last_err = out.get("error") if out else "Hasil kosong"
        if attempt < MAX_EVAL_ATTEMPTS:
            time.sleep(RETRY_DELAY)

    base = dict(state) if state else {}
    base["question"] = question

    if not out or "error" in out:
        msg = (
            f"Evaluasi gagal setelah {MAX_EVAL_ATTEMPTS} percobaan: {last_err}\n"
            "Pertanyaan TIDAK dianggap selesai — silakan coba lagi."
        )
        return msg, "", _empty_judge_df(), None, f"**Gagal:** {last_err}", question, base

    answer = out["answer"]
    request_id = out["request_id"]
    retrieval_context = out["retrieval_context"]
    result = out["result"]

    rec = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "request_id": request_id,
        "answer": answer,
        "retrieval_context_preview": (retrieval_context or "")[:2000],
    }
    rec.update(
        {
            "akurasi": result.get("akurasi"),
            "kelengkapan": result.get("kelengkapan"),
            "kesesuaian": result.get("kesesuaian"),
            "total": result.get("total"),
            "label": result.get("label"),
            "kesimpulan": result.get("kesimpulan"),
        }
    )
    save_eval(rec)
    DONE.add(question)

    fastapi_text = f"Jawaban (FastAPI):\n{answer}\n"
    trace_text = f"Konteks retrieval (trace):\n{retrieval_context}"
    judge_df = _judge_dataframe(result)
    judge_plot = _judge_plot_df(result)
    judge_summary = _judge_summary(result)

    nxt = question
    base.update(
        {
            "question": question,
            "answer": answer,
            "request_id": request_id,
            "retrieval_context": retrieval_context,
        }
    )
    if base.get("questions"):
        nxt, base = advance_sequence(base)

    return fastapi_text, trace_text, judge_df, judge_plot, judge_summary, nxt, base


def main():
    if not os.getenv("CHATOPA_URL") or not os.getenv("CHATOPA_API_KEY"):
        print("Konfigurasi CHATOPA_URL / CHATOPA_API_KEY belum lengkap di .env.")
    if not os.getenv("BASE_URL") or not os.getenv("OPENAI_API_KEY"):
        print("Konfigurasi BASE_URL / OPENAI_API_KEY belum lengkap di .env.")

    load_files()
    load_done()

    with gr.Blocks(title="Dashboard Evaluasi Chatbot RAG") as demo:
        with gr.Tabs():
            with gr.Tab("Evaluasi"):
                with gr.Row():
                    file_dd = gr.Dropdown(
                        label="Pilih Dokumen",
                        choices=[_file_label(i) for i in range(len(FILES))],
                    )
                    auto_q = gr.Textbox(
                        label="Pertanyaan",
                        interactive=False,
                        lines=2,
                    )
                    manual_in = gr.Textbox(
                        label=" ketik manual",
                        placeholder="Contoh: Apa perbedaan sawit dan kelapa?",
                        lines=2,
                    )
                    btn = gr.Button("Evaluasi", variant="primary")

                state = gr.State()

                with gr.Row():
                    with gr.Column():
                        fastapi_out = gr.Textbox(
                            label="Jawaban FastAPI (Chatbot)",
                            lines=14,
                            interactive=False,
                        )
                    with gr.Column():
                        trace_out = gr.Textbox(
                            label="Konteks Retrieval (Phoenix)",
                            lines=14,
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column():
                        judge_table = gr.Dataframe(
                            headers=["Kriteria", "Skor", "Alasan"],
                            label="Hasil Judge",
                            interactive=False,
                        )
                    with gr.Column():
                        judge_plot = gr.BarPlot(
                            x="Kriteria",
                            y="Skor",
                            title="Skor per Kriteria",
                            height=220,
                        )
                        judge_summary = gr.Markdown()

                file_dd.change(
                    on_file_change, inputs=[file_dd, state], outputs=[auto_q, state]
                )

                btn.click(
                    clear_all,
                    inputs=[],
                    outputs=[
                        fastapi_out,
                        trace_out,
                        judge_table,
                        judge_plot,
                        judge_summary,
                    ],
                ).then(
                    run_eval,
                    inputs=[manual_in, state],
                    outputs=[
                        fastapi_out,
                        trace_out,
                        judge_table,
                        judge_plot,
                        judge_summary,
                        auto_q,
                        state,
                    ],
                )

            with gr.Tab("Rekapitulasi Batch"):
                recap_table = gr.Dataframe(
                    headers=[
                        "Timestamp",
                        "Pertanyaan",
                        "Total",
                        "Label",
                        "Request ID",
                    ],
                    label="Hasil evaluasi ",
                    interactive=False,
                )
                refresh_btn = gr.Button("Muat Ulang", variant="secondary")
                refresh_btn.click(
                    load_recap, inputs=[], outputs=[recap_table]
                )

    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
