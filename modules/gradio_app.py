import asyncio
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
from modules.questions_generator import iter_generate_questions_from_files

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


def _file_label(i: int, files=None) -> str:
    files = files if files is not None else FILES
    return f"{i + 1}. {files[i]['source']} ({len(files[i]['questions'])} soal)"


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


AUTO_MAX_ATTEMPTS = 10


def _eval_with_retry(question: str, attempts: int = MAX_EVAL_ATTEMPTS):
    """Retry pipeline sampai semua step keluar hasil.
    Return ("ok", out) atau ("fail", last_err)."""
    last_err = None
    for attempt in range(1, attempts + 1):
        print(f"[EVAL] percobaan {attempt}/{attempts}: {question[:60]}")
        out = _try_full_eval(question)
        if out and "error" not in out:
            return "ok", out
        last_err = out.get("error") if out else "Hasil kosong"
        if attempt < attempts:
            time.sleep(RETRY_DELAY)
    return "fail", last_err


def _build_rec(question: str, out: dict) -> dict:
    result = out["result"]
    rec = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "request_id": out["request_id"],
        "answer": out["answer"],
        "retrieval_context_preview": (out["retrieval_context"] or "")[:2000],
        "akurasi": result.get("akurasi"),
        "kelengkapan": result.get("kelengkapan"),
        "kesesuaian": result.get("kesesuaian"),
        "total": result.get("total"),
        "label": result.get("label"),
        "kesimpulan": result.get("kesimpulan"),
    }
    return rec


def _build_outputs(out: dict) -> tuple:
    fastapi_text = f"Jawaban (FastAPI):\n{out['answer']}\n"
    trace_text = f"Konteks retrieval (trace):\n{out['retrieval_context']}"
    judge_df = _judge_dataframe(out["result"])
    judge_plot = _judge_plot_df(out["result"])
    judge_summary = _judge_summary(out["result"])
    return fastapi_text, trace_text, judge_df, judge_plot, judge_summary


def run_eval(manual: str, state: dict) -> tuple:
    """Evaluasi penuh dengan retry otomatis: POST -> trace -> judge diulang
    sampai semua step keluar hasil. Hanya soal yang BERHASIL semua step yang
    disimpan, ditandai selesai, dan dilanjutkan ke soal berikutnya."""
    question = (manual or "").strip() or (state or {}).get("question", "").strip()
    if not question:
        base = {"question": ""}
        return "Pertanyaan kosong.", "", _empty_judge_df(), None, "", "", base

    base = dict(state) if state else {}
    base["question"] = question

    status, payload = _eval_with_retry(question)
    if status == "fail":
        msg = (
            f"Evaluasi gagal setelah {MAX_EVAL_ATTEMPTS} percobaan: {payload}\n"
            "Pertanyaan TIDAK dianggap selesai — silakan coba lagi."
        )
        return msg, "", _empty_judge_df(), None, f"**Gagal:** {payload}", question, base

    out = payload
    save_eval(_build_rec(question, out))
    DONE.add(question)

    fastapi_text, trace_text, judge_df, judge_plot, judge_summary = _build_outputs(out)

    nxt = question
    base.update(
        {
            "question": question,
            "answer": out["answer"],
            "request_id": out["request_id"],
            "retrieval_context": out["retrieval_context"],
        }
    )
    if base.get("questions"):
        nxt, base = advance_sequence(base)

    return fastapi_text, trace_text, judge_df, judge_plot, judge_summary, nxt, base


def auto_eval_all(start_label, state: dict, progress=gr.Progress(), files=None,
                  sync_dropdown=True):
    """Auto evaluasi semua file berurutan tanpa klik manual.

    Untuk tiap file: tiap pertanyaan yang belum dinilai dievaluasi penuh
    (POST -> trace -> judge) dengan retry sampai semua step keluar hasil.
    Baru setelah soal berhasil, lanjut ke soal berikutnya. Setelah seluruh
    soal dalam file tuntas, otomatis pindah ke file berikutnya. Jika suatu
    soal tidak berhasil dinilai setelah banyak percobaan, proses DIHENTIKAN
    agar tidak ada step/pertanyaan yang tertinggal.

    sync_dropdown=False => slot dropdown (index 6) tidak diganti nilainya
    (dipakai alur upload file, supaya dropdown "Pilih Dokumen" tetap aman)."""
    state = dict(state) if state else {}
    if files is None:
        files = FILES

    start_idx = _file_index(start_label)
    if start_idx is None:
        start_idx = _file_index(state.get("file_label"))
    if start_idx is None:
        start_idx = 0
    start_idx = max(0, min(start_idx, len(files) - 1))

    dd_out = gr.update() if not sync_dropdown else _file_label(start_idx, files)

    empty_df = _empty_judge_df()
    done_count = 0
    failed_count = 0
    fastapi_text, trace_text = "", ""
    judge_df, judge_plot = empty_df, None
    judge_summary = ""
    label = _file_label(start_idx, files)
    last_ui = ("", "", empty_df, None, "", "", dd_out, dict(state),
               f"Memulai auto evaluasi dari {_file_label(start_idx, files)} ...")
    yield last_ui

    total_tasks = sum(
        1
        for fi in range(start_idx, len(files))
        for q in files[fi]["questions"]
        if q not in DONE
    )
    processed = 0

    for fi in range(start_idx, len(files)):
        f = files[fi]
        qs = f["questions"]
        label = _file_label(fi, files)
        state["file_label"] = label
        state["questions"] = qs

        for qi, question in enumerate(qs):
            state["q_index"] = qi
            state["question"] = question
            if question in DONE:
                continue

            status, payload = _eval_with_retry(question, attempts=AUTO_MAX_ATTEMPTS)
            processed += 1
            progress(
                processed / max(total_tasks, 1),
                desc=f"{label} | soal {qi + 1}/{len(qs)} | {question[:45]}",
            )

            if status == "ok":
                out = payload
                save_eval(_build_rec(question, out))
                DONE.add(question)
                done_count += 1
                fastapi_text, trace_text, judge_df, judge_plot, judge_summary = _build_outputs(out)
                status_msg = (
                    f"Soal {qi + 1}/{len(qs)} berhasil dinilai — {label}.\n"
                    f"Soal selesai: {done_count}. Lanjut ke berikutnya..."
                )
            else:
                failed_count += 1
                fastapi_text, trace_text, judge_df, judge_plot = "", "", empty_df, None
                judge_summary = f"**Gagal dinilai:** {payload}"
                status_msg = (
                    f"Soal gagal dinilai setelah {AUTO_MAX_ATTEMPTS} percobaan.\n"
                    f"Pertanyaan: {question}\n"
                    f"Error: {payload}\n\n"
                    "Proses dihentikan sampai soal ini berhasil. Cek tracing/"
                    "retrieval aktif, lalu klik Auto Evaluasi lagi."
                )
                state["question"] = question
                yield (
                    fastapi_text,
                    trace_text,
                    judge_df,
                    judge_plot,
                    judge_summary,
                    question,
                    (gr.update() if not sync_dropdown else label),
                    dict(state),
                    status_msg,
                )
                return

            state["question"] = question
            state["q_index"] = qi
            yield (
                fastapi_text,
                trace_text,
                judge_df,
                judge_plot,
                judge_summary,
                question,
                (gr.update() if not sync_dropdown else label),
                dict(state),
                status_msg,
            )

    final_msg = (
        f"Auto evaluasi selesai.\n"
        f"- Soal berhasil dinilai: {done_count}\n"
        f"- Soal gagal: {failed_count}"
    )
    last_ui = (
        fastapi_text,
        trace_text,
        judge_df,
        judge_plot,
        judge_summary,
        "",
        (gr.update() if not sync_dropdown else label),
        dict(state),
        final_msg,
    )
    yield last_ui


def _resolve_uploaded_path(uploaded) -> str:
    """gr.File bisa mengembalikan str path atau objek FileData."""
    if uploaded is None:
        return None
    if isinstance(uploaded, str):
        return uploaded
    return (
        getattr(uploaded, "path", None)
        or getattr(uploaded, "name", None)
        or str(uploaded)
    )


def _post_retry(question, attempts):
    """Retry POST ke chatbot sampai jawaban keluar.
    Return ("ok", (answer, request_id)) atau ("fail", last_err)."""
    last_err = None
    for attempt in range(1, attempts + 1):
        print(f"[POST] percobaan {attempt}/{attempts}: {question[:60]}")
        try:
            answer, request_id = get_answer(question)
            if answer:
                return "ok", (answer, request_id)
            last_err = "POST tidak mengembalikan jawaban."
        except Exception as e:
            print(f"[POST] error: {e}")
            last_err = f"POST ke chatbot gagal: {e}"
        if attempt < attempts:
            time.sleep(RETRY_DELAY)
    return "fail", last_err


def _trace_retry(request_id, question, attempts):
    """Retry ambil konteks retrieval dari Phoenix sampai keluar.
    Return ("ok", context_str) atau ("fail", last_err)."""
    last_err = None
    for attempt in range(1, attempts + 1):
        print(f"[TRACE] percobaan {attempt}/{attempts}: {question[:60]}")
        try:
            ctx = get_retrieval_context(request_id, question)
            if ctx:
                return "ok", ctx
            last_err = "Konteks retrieval tidak ditemukan."
        except Exception as e:
            print(f"[TRACE] error: {e}")
            last_err = f"Ambil konteks retrieval gagal: {e}"
        if attempt < attempts:
            time.sleep(RETRY_DELAY)
    return "fail", last_err


def _judge_retry(question, answer, retrieval_context, attempts):
    """Retry judge sampai hasil keluar.
    Return ("ok", result_dict) atau ("fail", last_err)."""
    last_err = None
    for attempt in range(1, attempts + 1):
        print(f"[JUDGE] percobaan {attempt}/{attempts}: {question[:60]}")
        try:
            result = judge_answer(question, answer, retrieval_context)
        except Exception as e:
            print(f"[JUDGE] error: {e}")
            last_err = f"Judge gagal: {e}"
            if attempt < attempts:
                time.sleep(RETRY_DELAY)
            continue
        if result and "error" not in result:
            return "ok", result
        last_err = (result or {}).get("error", "Judge tidak mengembalikan hasil.")
        if attempt < attempts:
            time.sleep(RETRY_DELAY)
    return "fail", last_err


def _resolve_uploaded_paths(uploaded):
    """Untuk file_count='multiple': normalisasi ke list path str."""
    if uploaded is None:
        return []
    if isinstance(uploaded, (list, tuple)):
        paths = []
        for item in uploaded:
            p = _resolve_uploaded_path(item)
            if p:
                paths.append(p)
        return paths
    p = _resolve_uploaded_path(uploaded)
    return [p] if p else []


def upload_pipeline(uploaded, state: dict, progress=gr.Progress()):
    """Tab Upload: upload beberapa file (.md/.txt/.pdf/.docx/.doc)
    -> generate pertanyaan PER-CHUNK -> langsung dinilai satu per satu
    (POST -> trace -> judge) tanpa menunggu semua pertanyaan jadi.

    Output tuple 8 elemen untuk komponen di tab Upload:
    (up_fastapi, up_trace, up_table, up_plot, up_summary, up_q, up_state, up_status)."""
    state = dict(state) if state else {}
    empty_df = _empty_judge_df()

    file_paths = _resolve_uploaded_paths(uploaded)
    if not file_paths:
        yield (
            "",
            "",
            empty_df,
            None,
            "",
            "",
            dict(state),
            "Belum ada file yang diupload. Upload dokumen (.md/.txt/.pdf/.docx/.doc) dulu.",
        )
        return

    names = ",\n".join(Path(p).name for p in file_paths)
    yield (
        "",
        "",
        empty_df,
        None,
        "",
        "",
        dict(state),
        f"Mulai proses dari:\n```\n{names}\n```",
    )

    done_count = 0
    failed_count = 0
    total_questions = 0
    total_files = 0
    last_fastapi, last_trace = "", ""
    last_df, last_plot, last_summary = empty_df, None, ""

    try:
        for source, chunk_idx, chunk_total, questions, item, items in (
            iter_generate_questions_from_files(file_paths)
        ):
            state["file_label"] = source
            if item["questions"] and item not in state.get("upload_items", []):
                state.setdefault("upload_items", []).append(item)
            state["questions"] = item["questions"]
            state["q_index"] = 0

            chunk_display = f"chunk {chunk_idx + 1}/{chunk_total}"

            for qi, question in enumerate(questions):
                state["question"] = question
                qpos = f"{qi + 1}/{len(questions)}"
                qinfo = f"({chunk_display}) — `{source}`"

                yield (
                    "",
                    "",
                    empty_df,
                    None,
                    "",
                    question,
                    dict(state),
                    f"Menilai pertanyaan {qpos} {qinfo}...",
                )

                status, payload = _post_retry(question, attempts=AUTO_MAX_ATTEMPTS)
                if status == "fail":
                    failed_count += 1
                    status_msg = (
                        f"Soal {qpos} gagal POST setelah {AUTO_MAX_ATTEMPTS} percobaan.\n"
                        f"Q: {question}\n"
                        f"Error: {payload}\n\n"
                        "Proses dihentikan sampai soal ini berhasil. Cek chatbot aktif, "
                        "lalu klik Generate → Evaluasi lagi."
                    )
                    yield (
                        last_fastapi,
                        last_trace,
                        last_df,
                        last_plot,
                        last_summary,
                        question,
                        dict(state),
                        status_msg,
                    )
                    return
                answer, request_id = payload
                fastapi_text = f"Jawaban (FastAPI):\n{answer}\n"

                yield (
                    fastapi_text,
                    last_trace,
                    last_df,
                    last_plot,
                    last_summary,
                    question,
                    dict(state),
                    f"Jawaban POST diterima ({qpos}) {qinfo}. Mengambil konteks dari Phoenix...",
                )

                status, payload = _trace_retry(
                    request_id, question, attempts=AUTO_MAX_ATTEMPTS
                )
                if status == "fail":
                    failed_count += 1
                    status_msg = (
                        f"Soal {qpos} gagal ambil konteks setelah {AUTO_MAX_ATTEMPTS} percobaan.\n"
                        f"Q: {question}\n"
                        f"Error: {payload}\n\n"
                        "Proses dihentikan sampai soal ini berhasil. Cek tracing/retrieval "
                        "aktif, lalu klik Generate → Evaluasi lagi."
                    )
                    yield (
                        fastapi_text,
                        last_trace,
                        last_df,
                        last_plot,
                        last_summary,
                        question,
                        dict(state),
                        status_msg,
                    )
                    return
                retrieval_context = payload
                trace_text = f"Konteks retrieval (trace):\n{retrieval_context}"

                yield (
                    fastapi_text,
                    trace_text,
                    last_df,
                    last_plot,
                    last_summary,
                    question,
                    dict(state),
                    f"Konteks retrieval diterima ({qpos}) {qinfo}. Menilai jawaban...",
                )

                status, payload = _judge_retry(
                    question, answer, retrieval_context, attempts=AUTO_MAX_ATTEMPTS
                )
                if status == "fail":
                    failed_count += 1
                    status_msg = (
                        f"Soal {qpos} gagal dinilai setelah {AUTO_MAX_ATTEMPTS} percobaan.\n"
                        f"Q: {question}\n"
                        f"Error: {payload}\n\n"
                        "Proses dihentikan sampai soal ini berhasil. Cek relay judge aktif, "
                        "lalu klik Generate → Evaluasi lagi."
                    )
                    yield (
                        fastapi_text,
                        trace_text,
                        last_df,
                        last_plot,
                        last_summary,
                        question,
                        dict(state),
                        status_msg,
                    )
                    return

                out = {
                    "answer": answer,
                    "request_id": request_id,
                    "retrieval_context": retrieval_context,
                    "result": payload,
                }
                save_eval(_build_rec(question, out))
                DONE.add(question)
                done_count += 1
                last_fastapi, last_trace, last_df, last_plot, last_summary = (
                    _build_outputs(out)
                )
                status_msg = (
                    f"Soal {done_count} berhasil dinilai ({qpos}) {qinfo}\n"
                    f"Q: {question}"
                )

                progress(
                    (done_count + failed_count) / max(total_questions, 1),
                    desc=f"{source} | {chunk_display} | {question[:45]}",
                )
                yield (
                    last_fastapi,
                    last_trace,
                    last_df,
                    last_plot,
                    last_summary,
                    question,
                    dict(state),
                    status_msg,
                )

            if item["questions"] and item not in state.get("upload_items", []):
                state.setdefault("upload_items", []).append(item)

            total_questions = sum(
                len(f["questions"]) for f in state.get("upload_items", [])
            )
            total_files = len(state.get("upload_items", []))

        state["question"] = ""
        final_msg = (
            f"Proses selesai.\n"
            f"- File diproses: {total_files}\n"
            f"- Pertanyaan berhasil dinilai: {done_count}\n"
            f"- Pertanyaan gagal: {failed_count}\n\n"
            "Hasil tersimpan di `output/questions_upload.json` (pertanyaan) "
            "dan `output/evaluations.jsonl` (penilaian)."
        )
        yield (
            last_fastapi,
            last_trace,
            last_df,
            last_plot,
            last_summary,
            "",
            dict(state),
            final_msg,
        )

    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield (
            last_fastapi,
            last_trace,
            last_df,
            last_plot,
            last_summary,
            "",
            dict(state),
            f"Generate gagal: {e}",
        )
        return


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
                    with gr.Group():
                        file_dd = gr.Dropdown(
                            label="Pilih Dokumen",
                            choices=[_file_label(i) for i in range(len(FILES))],
                            scale=1,
                            min_width=180,
                        )
                    with gr.Group():
                        auto_q = gr.Textbox(
                            label="Pertanyaan",
                            interactive=False,
                            lines=2,
                            scale=1,
                            min_width=240,
                        )
                        with gr.Row():
                            auto_btn = gr.Button(
                                "Auto Evaluasi", variant="secondary", scale=1, min_width=140
                            )
                            stop_btn = gr.Button(
                                "Stop", variant="stop", scale=1, min_width=140
                            )
                    with gr.Group():
                        manual_in = gr.Textbox(
                            label=" ketik manual",
                            placeholder="Contoh: Apa perbedaan sawit dan kelapa?",
                            lines=2,
                            scale=1,
                            min_width=240,
                        )
                        btn = gr.Button(
                            "Evaluasi", variant="primary", scale=1, min_width=140
                        )

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

                auto_status = gr.Markdown()

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

                auto_event = auto_btn.click(
                    auto_eval_all,
                    inputs=[file_dd, state],
                    outputs=[
                        fastapi_out,
                        trace_out,
                        judge_table,
                        judge_plot,
                        judge_summary,
                        auto_q,
                        file_dd,
                        state,
                        auto_status,
                    ],
                )
                stop_btn.click(
                    fn=lambda: "⏹ Auto evaluasi dihentikan.",
                    inputs=[],
                    outputs=[auto_status],
                    cancels=[auto_event],
                )

            with gr.Tab("Upload Dokumen"):
                with gr.Row():
                    with gr.Group():
                        upload_file = gr.File(
                            label="Upload dokumen (md, txt, pdf, docx, doc)",
                            file_count="multiple",
                            file_types=[".md", ".txt", ".pdf", ".docx", ".doc"],
                            scale=1,
                            min_width=240,
                        )
                        with gr.Row():
                            pipeline_btn = gr.Button(
                                "Generate → Evaluasi",
                                variant="primary",
                                scale=1,
                                min_width=140,
                            )
                            up_stop_btn = gr.Button(
                                "Stop", variant="stop", scale=1, min_width=140
                            )

                up_state = gr.State()

                with gr.Row():
                    with gr.Column():
                        up_fastapi = gr.Textbox(
                            label="Jawaban FastAPI (Chatbot)",
                            lines=14,
                            interactive=False,
                        )
                    with gr.Column():
                        up_trace = gr.Textbox(
                            label="Konteks Retrieval (Phoenix)",
                            lines=14,
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column():
                        up_table = gr.Dataframe(
                            headers=["Kriteria", "Skor", "Alasan"],
                            label="Hasil Judge",
                            interactive=False,
                        )
                    with gr.Column():
                        up_plot = gr.BarPlot(
                            x="Kriteria",
                            y="Skor",
                            title="Skor per Kriteria",
                            height=220,
                        )
                        up_summary = gr.Markdown()

                up_q = gr.Textbox(
                    label="Pertanyaan",
                    interactive=False,
                    lines=2,
                )
                up_status = gr.Markdown()

                pipeline_event = pipeline_btn.click(
                    upload_pipeline,
                    inputs=[upload_file, up_state],
                    outputs=[
                        up_fastapi,
                        up_trace,
                        up_table,
                        up_plot,
                        up_summary,
                        up_q,
                        up_state,
                        up_status,
                    ],
                )
                up_stop_btn.click(
                    fn=lambda: "⏹ Proses dihentikan.",
                    inputs=[],
                    outputs=[up_status],
                    cancels=[pipeline_event],
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
