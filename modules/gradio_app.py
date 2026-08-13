import asyncio
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.judge import get_answer, judge_answer
from modules.phoenix_extractor import get_retrieval_context
from modules.questions_generator import iter_generate_questions_from_files
from modules.doc_chat import ask as doc_ask
from modules.doc_chat import index_documents as doc_index
from modules import doc_chat as doc_chat_mod
from modules.pipeline_watcher import evaluate_new_traces
from modules import tracing_control
from modules import pipeline_judge

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "output" / "questions.json"
EVAL_PATH = ROOT / "output" / "evaluations.jsonl"
PIPELINE_EVAL_PATH = ROOT / "output" / "evaluations_pipeline.jsonl"

FILES = []
DONE = set()
PIPELINE_DONE = set()

SKIP_JUDGE = False  # Judge aktif (mode normal): POST -> GET -> judge.


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
            if rec.get("type") == "chat_doc":
                continue
            if rec.get("question") and rec.get("total") is not None and not rec.get("error"):
                DONE.add(rec["question"])


def save_eval(rec: dict):
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_pipeline_done():
    global PIPELINE_DONE
    PIPELINE_DONE = set()
    if not PIPELINE_EVAL_PATH.exists():
        return
    with open(PIPELINE_EVAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("trace_id") and rec.get("total") is not None and not rec.get("error"):
                PIPELINE_DONE.add(rec["trace_id"])


def save_pipeline_eval(rec: dict):
    PIPELINE_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_EVAL_PATH, "a", encoding="utf-8") as f:
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
                        "Tipe": "Chat Dokumen" if rec.get("type") == "chat_doc" else "Evaluasi",
                        "Timestamp": rec.get("timestamp"),
                        "Pertanyaan": rec.get("question"),
                        "Dokumen Acuan": rec.get("n_acuan"),
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


def _first_label(file_label):
    """Dropdown multiselect bisa berupa list -> pakai label pertama."""
    if isinstance(file_label, (list, tuple)):
        return file_label[0] if file_label else None
    return file_label


def on_file_change(file_label, state: dict) -> tuple:
    """Pilih dokumen -> muat pertanyaan pertama yang belum dievaluasi."""
    state = dict(state) if state else {}
    file_label = _first_label(file_label)
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
    return [], [], empty_df, None, ""


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

    if SKIP_JUDGE:
        result = _skip_judge_result(question, retrieval_context)
    else:
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


def _count_documents(retrieval_context: str) -> int:
    """Jumlah dokumen acuan unik dari konteks retrieval penuh.
    Format blok: '[1] Source: (...)' dst."""
    if not retrieval_context:
        return 0
    indexes = re.findall(r"\[(\d+)\]\s*Source:", retrieval_context)
    return len(set(int(i) for i in indexes))


def _build_rec(question: str, out: dict) -> dict:
    result = out["result"]
    rec = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "request_id": out["request_id"],
        "answer": out["answer"],
        "retrieval_context_preview": (out["retrieval_context"] or "")[:2000],
        "n_acuan": _count_documents(out.get("retrieval_context") or ""),
        "akurasi": result.get("akurasi"),
        "kelengkapan": result.get("kelengkapan"),
        "kesesuaian": result.get("kesesuaian"),
        "total": result.get("total"),
        "label": result.get("label"),
        "kesimpulan": result.get("kesimpulan"),
    }
    return rec


def _skip_judge_result(question: str, context):
    """Stub hasil saat SKIP_JUDGE aktif: judge tidak dipanggil,
    tapi UI tetap mendapat dict bertanda 'skipped' untuk disembunyikan."""
    return {
        "skipped": True,
        "akurasi": {"skor": None, "alasan": "Judge dilewati (mode uji CSS)."},
        "kelengkapan": {"skor": None, "alasan": "Judge dilewati (mode uji CSS)."},
        "kesesuaian": {"skor": None, "alasan": "Judge dilewati (mode uji CSS)."},
        "total": None,
        "label": "SKIP",
        "kesimpulan": (
            "Judge dinonaktifkan — hanya menampilkan jawaban POST & "
            "konteks retrieval untuk uji tampilan CSS."
        ),
    }


def _is_skipped(out: dict) -> bool:
    return bool((out.get("result") or {}).get("skipped"))


def _qa_messages(question, answer):
    """Pertanyaan+jawaban jadi pesan chat (format Gradio 6 Chatbot)."""
    if not answer:
        return []
    return [
        {"role": "user", "content": question or ""},
        {"role": "assistant", "content": answer},
    ]


def _clean_retrieval(context: str) -> str:
    """Bersihkan metadata header: Knowledge Context, baris '[N] Source:'
    dan label 'Content:' — sisakan isi dokumen (blok sitasi tetap)."""
    if not context:
        return context
    c = re.sub(r"^\s*-\s*Knowledge Context:\s*\n?", "", context)
    c = re.sub(r"^\s*\[\d+\]\s*Source:[^\n]*\n?", "", c, flags=re.M)
    c = re.sub(r"^\s*Content:\s*\n?", "", c, flags=re.M)
    return c.strip()


def _trace_msgs(text: str) -> list:
    """Konteks retrieval jadi pesan Chatbot (format Gradio 6)."""
    if not text:
        return []
    return [{"role": "assistant", "content": text}]


def _build_outputs(out: dict, question: str = "") -> tuple:
    qa_msgs = _qa_messages(question, out.get("answer"))
    trace_text = _trace_msgs(
        f"Konteks retrieval (trace):\n{_clean_retrieval(out.get('retrieval_context') or '')}"
    )
    judge_df = _judge_dataframe(out["result"])
    judge_plot = _judge_plot_df(out["result"])
    judge_summary = _judge_summary(out["result"])
    return qa_msgs, trace_text, judge_df, judge_plot, judge_summary


def run_eval(manual: str, state: dict) -> tuple:
    """Evaluasi penuh dengan retry otomatis: POST -> trace -> judge diulang
    sampai semua step keluar hasil. Hanya soal yang BERHASIL semua step yang
    disimpan, ditandai selesai, dan dilanjutkan ke soal berikutnya."""
    question = (manual or "").strip() or (state or {}).get("question", "").strip()
    if not question:
        base = {"question": ""}
        return [], [], _empty_judge_df(), None, "Pertanyaan kosong.", "", base

    base = dict(state) if state else {}
    base["question"] = question

    status, payload = _eval_with_retry(question)
    if status == "fail":
        msg = (
            f"Evaluasi gagal setelah {MAX_EVAL_ATTEMPTS} percobaan: {payload}\n"
            "Pertanyaan TIDAK dianggap selesai — silakan coba lagi."
        )
        return [], [], _empty_judge_df(), None, f"**Gagal:** {payload}", question, base

    out = payload
    if not _is_skipped(out):
        save_eval(_build_rec(question, out))
        DONE.add(question)

    fastapi_text, trace_text, judge_df, judge_plot, judge_summary = _build_outputs(out, question)

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

    if isinstance(start_label, (list, tuple)):
        labels = list(start_label)
    elif start_label:
        labels = [start_label]
    else:
        labels = [state.get("file_label")]
    labels = [l for l in labels if l]

    if labels:
        indexes = sorted({_file_index(l) for l in labels if _file_index(l) is not None})
        if not indexes:
            indexes = [0]
    else:
        indexes = list(range(len(files)))
    indexes = [max(0, min(i, len(files) - 1)) for i in indexes]

    start_idx = indexes[0]
    dd_out = gr.update() if not sync_dropdown else _file_label(start_idx, files)

    empty_df = _empty_judge_df()
    done_count = 0
    failed_count = 0
    fastapi_msgs, trace_text = [], []
    judge_df, judge_plot = empty_df, None
    judge_summary = ""
    label = _file_label(start_idx, files)
    last_ui = ([], "", empty_df, None, "", "", dd_out, dict(state),
               f"Memulai auto evaluasi dari {_file_label(start_idx, files)} ...")
    yield last_ui

    total_tasks = sum(
        1
        for fi in indexes
        for q in files[fi]["questions"]
        if q not in DONE
    )
    processed = 0

    for fi in indexes:
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
                if not _is_skipped(out):
                    save_eval(_build_rec(question, out))
                    DONE.add(question)
                done_count += 1
                fastapi_msgs, trace_text, judge_df, judge_plot, judge_summary = _build_outputs(out, question)
                status_msg = (
                    f"Soal {qi + 1}/{len(qs)} berhasil dinilai — {label}.\n"
                    f"Soal selesai: {done_count}. Lanjut ke berikutnya..."
                )
            else:
                failed_count += 1
                fastapi_msgs, trace_text, judge_df, judge_plot = [], [], empty_df, None
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
                    fastapi_msgs,
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
                fastapi_msgs,
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
        fastapi_msgs,
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
    if SKIP_JUDGE:
        return "ok", _skip_judge_result(question, retrieval_context)
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
            [],
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
        [],
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
    last_fastapi, last_trace = [], []
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
                    [],
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
                fastapi_text = _qa_messages(question, answer)

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
                trace_text = _trace_msgs(f"Konteks retrieval (trace):\n{_clean_retrieval(retrieval_context)}")

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
                if not _is_skipped(out):
                    save_eval(_build_rec(question, out))
                    DONE.add(question)
                done_count += 1
                last_fastapi, last_trace, last_df, last_plot, last_summary = (
                    _build_outputs(out, question)
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


_INDEXED_SIG = None


def _chat_signature(paths):
    sig = []
    for p in paths:
        try:
            st = os.stat(p)
            sig.append((p, st.st_size, st.st_mtime))
        except OSError:
            sig.append((p, 0, 0))
    return tuple(sorted(sig))


def _chat_ensure_indexed(uploaded):
    """Index dokumen hanya jika set file berubah. Return (paths, status_msg)."""
    global _INDEXED_SIG
    paths = _resolve_uploaded_paths(uploaded)
    if not paths:
        return [], "Belum ada dokumen yang diupload. Upload file dulu."
    sig = _chat_signature(paths)
    if sig == _INDEXED_SIG:
        return paths, ""
    n = doc_index(paths)
    if n == 0:
        return [], "Tidak ada chunk yang bisa diindeks dari file tersebut."
    _INDEXED_SIG = sig
    return paths, f"{n} chunk diindeks dari {len(paths)} file."


def _messages_to_pairs(history):
    """Konversi list dict messages (format Gradio 6) -> list [user, assistant]."""
    pairs = []
    pending = None
    for m in history or []:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content", "")
        elif isinstance(m, (list, tuple)) and len(m) >= 1:
            role = "user"
            content = m[0]
        else:
            continue
        if role == "user":
            if pending is not None:
                pairs.append([pending, ""])
            pending = content
        else:
            pairs.append([pending if pending is not None else "", content])
            pending = None
    if pending is not None:
        pairs.append([pending, ""])
    return pairs


def _pairs_to_messages(pairs):
    """Konversi list [user, assistant] -> list dict {role, content}."""
    out = []
    for user, assistant in pairs or []:
        out.append({"role": "user", "content": user})
        out.append({"role": "assistant", "content": assistant})
    return out


def _chat_format_sources(hits):
    if not hits:
        return "(tidak ada sumber — dijawab dari pengetahuan umum)"
    lines = []
    for i, h in enumerate(hits):
        snippet = h["text"].replace("\n", " ").strip()[:180]
        lines.append(f"**[{i + 1}] {h['source']}** (skor {h['score']})\n{snippet}")
    return "\n\n".join(lines)


def chat_index(uploaded):
    paths, msg = _chat_ensure_indexed(uploaded)
    return msg


def _chat_context(hits):
    """Format hits retrieval lokal -> string konteks [1]..[n] untuk judge."""
    if not hits:
        return None
    return "\n\n".join(
        f"[{i + 1}] Sumber: {h['source']} (skor {h['score']})\n{h['text']}"
        for i, h in enumerate(hits)
    )


def _chat_judge_retry(question, answer, context, attempts):
    """Judge jawaban chat dengan retry. Return ('ok', result) / ('fail', pesan_error)."""
    if SKIP_JUDGE:
        return "ok", _skip_judge_result(question, context)
    last_err = ""
    for attempt in range(1, attempts + 1):
        try:
            result = judge_answer(question, answer, context, domain="dokumen yang diupload")
            if result and "error" not in result:
                return "ok", result
            last_err = (result or {}).get("error", "Judge tidak mengembalikan hasil.")
        except Exception as e:
            last_err = str(e)
            print(f"[CHAT-JUDGE] percobaan {attempt}/{attempts}: {e}")
        time.sleep(RETRY_DELAY)
    return "fail", last_err or "Judge gagal setelah beberapa percobaan."


def _chat_judge_placeholder(message):
    """Tabel placeholder utk penilaian yang tidak dapat dijalankan."""
    return pd.DataFrame(
        [["Judge", "—", message]],
        columns=["Kriteria", "Skor", "Alasan"],
    )


def _chat_judge_df(result):
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


def _save_chat_eval(question, answer, hits, result):
    rec = {
        "type": "chat_doc",
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "n_acuan": len(hits) if hits else 0,
        "sumber": ", ".join(h["source"] for h in hits) if hits else "",
        "akurasi": result.get("akurasi", {}).get("skor"),
        "kelengkapan": result.get("kelengkapan", {}).get("skor"),
        "kesesuaian": result.get("kesesuaian", {}).get("skor"),
        "total": result.get("total"),
        "label": result.get("label"),
        "kesimpulan": result.get("kesimpulan"),
        "request_id": "",
    }
    save_eval(rec)


def chat_answer(message, history, uploaded):
    pairs = _messages_to_pairs(history)
    empty_df = _empty_judge_df()
    if not message or not message.strip():
        return _pairs_to_messages(pairs), "", empty_df, ""
    paths, status = _chat_ensure_indexed(uploaded)
    try:
        answer, hits = doc_ask(message.strip(), pairs)
    except Exception as e:
        pairs.append([message, f"Gagal menjawab: {e}"])
        return _pairs_to_messages(pairs), "", empty_df, status
    pairs.append([message, answer])

    context = _chat_context(hits)
    if context:
        judge_status, judge_result = _chat_judge_retry(
            message.strip(), answer, context, attempts=3
        )
        if judge_status == "ok":
            judge_df = _chat_judge_df(judge_result)
            try:
                if not judge_result.get("skipped"):
                    _save_chat_eval(message.strip(), answer, hits, judge_result)
            except Exception as e:
                print(f"[CHAT-JUDGE] gagal simpan rekap: {e}")
        else:
            judge_df = _chat_judge_placeholder(
                f"Judge gagal dijalankan: {judge_result}"
            )
    else:
        judge_df = _chat_judge_placeholder(
            "Tidak dapat dinilai: tidak ada sumber retrieval (pertanyaan di luar dokumen)."
        )
    return _pairs_to_messages(pairs), _chat_format_sources(hits), judge_df, status


def chat_reset():
    return [], "", _empty_judge_df(), "Chat dibersihkan."


def load_pipeline_recap() -> pd.DataFrame:
    """Baca output/evaluations_pipeline.jsonl -> DataFrame."""
    rows = []
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
                rows.append(
                    {
                        "Timestamp": rec.get("timestamp"),
                        "Pertanyaan": rec.get("question"),
                        "Intent & Understanding": (rec.get("intent_understanding") or {}).get("skor"),
                        "Query Expansion": (rec.get("query_expansion") or {}).get("skor"),
                        "Reasoning": (rec.get("reasoning") or {}).get("skor"),
                        "Memory Continuity": (rec.get("memory_continuity") or {}).get("skor"),
                        "Total": rec.get("total"),
                        "Trace ID": rec.get("trace_id"),
                    }
                )
    return pd.DataFrame(rows)


def _watcher_loop(stop_event, interval: float = 60.0, hours: int = 24):
    """Loop background: evaluasi trace staging baru selama app terbuka."""
    print("[WATCHER] Monitoring trace staging dimulai.")
    while not stop_event.is_set():
        try:
            summary = evaluate_new_traces(hours=hours)
            if summary["ok"]:
                print(
                    f"[WATCHER] Evaluasi baru: ok={summary['ok']} "
                    f"gagal={summary['gagal']} (total {summary['total']})"
                )
        except Exception as e:
            print(f"[WATCHER] Error: {e}")
        stop_event.wait(interval)
    print("[WATCHER] Monitoring dihentikan.")


WATCHER_STOP = threading.Event()
WATCHER_STARTED = False
WATCHER_THREAD = None


def watcher_status() -> str:
    watcher = (
        "**aktif** — menilai trace staging baru secara berkala."
        if WATCHER_STARTED
        else "**berhenti** — tekan Start Watcher untuk mulai menilai trace staging baru."
    )
    return f"Watcher {watcher}\n\n{tracing_control.status()}"


def watcher_start() -> str:
    global WATCHER_STARTED, WATCHER_THREAD, WATCHER_STOP
    if WATCHER_STARTED:
        return watcher_status()
    tracing_control.enable()
    pipeline_judge.reset_tracing()
    doc_chat_mod.reset_tracing()
    WATCHER_STOP = threading.Event()
    WATCHER_THREAD = threading.Thread(
        target=_watcher_loop,
        args=(WATCHER_STOP,),
        daemon=True,
    )
    WATCHER_THREAD.start()
    WATCHER_STARTED = True
    return watcher_status()


def watcher_stop() -> str:
    global WATCHER_STARTED, WATCHER_THREAD
    if not WATCHER_STARTED:
        return watcher_status()
    WATCHER_STOP.set()
    WATCHER_STARTED = False
    WATCHER_THREAD = None
    tracing_control.disable()
    return watcher_status()



def main():
    if not os.getenv("CHATOPA_URL") or not os.getenv("CHATOPA_API_KEY"):
        print("Konfigurasi CHATOPA_URL / CHATOPA_API_KEY belum lengkap di .env.")
    if not os.getenv("BASE_URL") or not os.getenv("OPENAI_API_KEY"):
        print("Konfigurasi BASE_URL / OPENAI_API_KEY belum lengkap di .env.")

    load_files()
    load_done()
    load_pipeline_done()

    with gr.Blocks(title="Dashboard Evaluasi Chatbot RAG") as demo:
        with gr.Tabs():
            with gr.Tab("Evaluasi"):
                with gr.Row():
                    with gr.Group():
                        file_dd = gr.Dropdown(
                            label="Pilih Dokumen",
                            choices=[_file_label(i) for i in range(len(FILES))],
                            multiselect=True,
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
                        fastapi_out = gr.Chatbot(
                            label="Percakapan (Pertanyaan → Jawaban)",
                            height=420,
                            elem_classes=["soft-box"],
                        )
                    with gr.Column():   
                        trace_out = gr.Chatbot(
                            label="Konteks Retrieval (Phoenix)",
                            height=420,
                            elem_classes=["soft-box"],
                        )

                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            judge_table = gr.Dataframe(
                                headers=["Kriteria", "Skor", "Alasan"],
                                label="Hasil Judge",
                                show_label=True,
                                interactive=False,
                                wrap=True,
                                line_breaks=True,
                                column_widths=[120, 60, 420],
                                max_height=320,
                            )
                        with gr.Column():
                            judge_plot = gr.BarPlot(
                                x="Kriteria",
                                y="Skor",
                                y_lim=[0, 10],
                                height=180,
                                sort="-y",
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

            with gr.Tab("Evaluasi Pipeline"):
                with gr.Row():
                    with gr.Group():
                        pipe_watch_status = gr.Markdown(watcher_status())
                    with gr.Group():
                        pipe_watch_start_btn = gr.Button(
                            "Start Watcher", variant="secondary", scale=1, min_width=140
                        )
                        pipe_watch_stop_btn = gr.Button(
                            "Stop Watcher", variant="stop", scale=1, min_width=140
                        )

                pipe_pipeline_recap = gr.Dataframe(
                    headers=[
                        "Timestamp",
                        "Pertanyaan",
                        "Intent & Understanding",
                        "Query Expansion",
                        "Reasoning",
                        "Memory Continuity",
                        "Total",
                        "Trace ID",
                    ],
                    label="Rekap Hasil Evaluasi Pipeline",
                    show_label=True,
                    interactive=False,
                    wrap=True,
                    line_breaks=True,
                    column_widths=[150, 360, 130, 120, 90, 130, 75, 240],
                    max_height=420,
                    value=load_pipeline_recap(),
                )
                pipe_pipeline_refresh_btn = gr.Button(
                    "Muat Ulang Hasil", variant="secondary"
                )

                pipe_watch_start_btn.click(
                    watcher_start, inputs=[], outputs=[pipe_watch_status]
                )
                pipe_watch_stop_btn.click(
                    watcher_stop, inputs=[], outputs=[pipe_watch_status]
                )
                pipe_pipeline_refresh_btn.click(
                    load_pipeline_recap, inputs=[], outputs=[pipe_pipeline_recap]
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
                        up_fastapi = gr.Chatbot(
                            label="Percakapan (Pertanyaan → Jawaban)",
                            height=420,
                            elem_classes=["soft-box"],
                        )
                    with gr.Column():
                        up_trace = gr.Chatbot(
                            label="Konteks Retrieval (Phoenix)",
                            height=420,
                            elem_classes=["soft-box"],
                        )

                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            up_table = gr.Dataframe(
                                headers=["Kriteria", "Skor", "Alasan"],
                                label="Hasil Judge",
                                show_label=True,
                                interactive=False,
                                wrap=True,
                                line_breaks=True,
                                column_widths=[120, 60, 420],
                                max_height=320,
                            )
                        with gr.Column():
                            up_plot = gr.BarPlot(
                                x="Kriteria",
                                y="Skor",
                                y_lim=[0, 10],
                                height=180,
                                sort="-y",
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
                    fn=lambda: "Proses dihentikan.",
                    inputs=[],
                    outputs=[up_status],
                    cancels=[pipeline_event],
                )

            with gr.Tab("Chat Dokumen"):
                with gr.Group():
                    chat_upload = gr.File(
                        label="Upload dokumen (md, txt, pdf, docx, doc) — langsung dipelajari otomatis",
                        file_count="multiple",
                        file_types=[".md", ".txt", ".pdf", ".docx", ".doc"],
                        scale=1,
                        min_width=240,
                    )
                    chat_index_status = gr.Markdown()

                chatbot = gr.Chatbot(label="Chat", height=420, layout="bubble")
                with gr.Row():
                    chat_in = gr.Textbox(
                        label="Upload dokumen, lalu tulis pertanyaan di sini",
                        placeholder="Contoh: apa isi utama dokumen ini?",
                        lines=2,
                        scale=3,
                        min_width=240,
                    )
                    chat_send = gr.Button(
                        "Kirim", variant="primary", scale=1, min_width=120
                    )
                    chat_reset_btn = gr.Button(
                        "Reset", variant="stop", scale=1, min_width=120
                    )
                with gr.Row():
                    with gr.Group():
                        chat_sources = gr.Markdown(label="Sumber")
                    with gr.Group():
                        chat_judge = gr.Dataframe(
                            headers=["Kriteria", "Skor", "Alasan"],
                            label="Penilaian",
                            show_label=True,
                            interactive=False,
                            wrap=True,
                            line_breaks=True,
                            column_widths=[140, 60, 420],
                            max_height=320,
                        )

                chat_upload.change(
                    chat_index, inputs=[chat_upload], outputs=[chat_index_status]
                )

                chat_send.click(
                    chat_answer,
                    inputs=[chat_in, chatbot, chat_upload],
                    outputs=[chatbot, chat_sources, chat_judge, chat_index_status],
                )
                chat_in.submit(
                    chat_answer,
                    inputs=[chat_in, chatbot, chat_upload],
                    outputs=[chatbot, chat_sources, chat_judge, chat_index_status],
                )
                chat_reset_btn.click(
                    chat_reset, inputs=[], outputs=[chatbot, chat_sources, chat_judge, chat_index_status]
                )

            with gr.Tab("Rekapitulasi Batch"):
                recap_table = gr.Dataframe(
                    headers=[
                        "Tipe",
                        "Timestamp",
                        "Pertanyaan",
                        "Dokumen Acuan",
                        "Total",
                        "Label",
                        "Request ID",
                    ],
                    label="Hasil Evaluasi",
                    show_label=True,
                    elem_classes=["recap-table"],
                    interactive=False,
                    wrap=True,
                    line_breaks=True,
                    column_widths=[60, 150, 360, 110, 75, 95, 240],
                    max_height=420,
                )
                refresh_btn = gr.Button("Muat Ulang", variant="secondary")
                refresh_btn.click(
                    load_recap, inputs=[], outputs=[recap_table]
                )

                pipe_recap_table = gr.Dataframe(
                    headers=[
                        "Timestamp",
                        "Pertanyaan",
                        "Intent & Understanding",
                        "Query Expansion",
                        "Reasoning",
                        "Memory Continuity",
                        "Total",
                        "Trace ID",
                    ],
                    label="Rekap Evaluasi Pipeline",
                    show_label=True,
                    elem_classes=["recap-table"],
                    interactive=False,
                    wrap=True,
                    line_breaks=True,
                    column_widths=[150, 360, 130, 120, 90, 130, 75, 240],
                    max_height=420,
                )
                pipe_refresh_btn = gr.Button("Muat Ulang Rekap Pipeline", variant="secondary")
                pipe_refresh_btn.click(
                    load_pipeline_recap, inputs=[], outputs=[pipe_recap_table]
                )

    _CSS = """
/* ==========================================================================
   1. DASAR KOMPONEN & ELEMEN FORM
   ========================================================================== */

/* Tombol utama */
button { 
  border-radius: 9px !important; 
  padding: 8px 24px !important; 
}

/* Card/Container tanpa sudut melengkung */
.gr-group { 
  padding: 14px !important; 
  border-radius: 0 !important;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important; 
}

/* Input, Textarea, dan Select umum */
input, textarea, select { 
  border-radius: 12px !important; 
}

/* Teks pada kotak non-interaktif (Readonly / Disabled): dibuat gelap agar kebaca */
textarea:disabled, input:disabled,
textarea[readonly], input[readonly] {
  color: rgba(51, 65, 85, 1) !important;
  -webkit-text-fill-color: rgba(51, 65, 85, 1) !important;
  font-weight: 400 !important;
  opacity: 1 !important;
}

/* Kotak Pertanyaan (auto-load): samakan dengan input lain sesuai tema */
textarea[data-testid="textbox"]:disabled,
textarea[data-testid="textbox"][readonly] {
  background-color: var(--input-background-fill) !important;
  color: var(--body-text-color) !important;
  -webkit-text-fill-color: var(--body-text-color) !important;
  border-color: var(--input-border-color) !important;
  opacity: 1 !important;
}

/* Hilangkan lapisan gradient penutup agar teks tidak tertutup */
.scroll-fade {
  background: transparent !important;
  opacity: 0 !important;
}

/* Hapus placeholder pada input pertanyaan manual */
textarea[placeholder="Contoh: Apa perbedaan sawit dan kelapa?"]::placeholder,
textarea[placeholder="Contoh: Apa perbedaan sawit dan kelapa?"]::-webkit-input-placeholder {
  color: transparent !important;
  opacity: 0 !important;
}

/* ==========================================================================
   2. LABEL & BADGE (Pilih Dokumen, Pertanyaan, Hasil Evaluasi, dll.)
   ========================================================================== */

:root {
  --block-label-text-color: rgba(30, 58, 138, 0.9) !important;
  --block-label-text-weight: 600 !important;
}

/* Label Komponen UI & Label Tabel Dataframe */
.header-row .label {
  background: var(--block-label-background-fill) !important;
  border-radius: var(--block-label-radius) !important;
  padding: var(--block-label-padding) !important;
  white-space: nowrap !important;
  flex: none !important;
}

.header-row .label p {
  color: var(--block-label-text-color) !important;
  font-weight: var(--block-label-text-weight) !important;
  font-size: var(--block-label-text-size) !important;
}

/* ==========================================================================
   3. CHAT BUBBLE & PERCAKAPAN
   ========================================================================== */

.bubble.user-row { 
  background-color: var(--color-neutral-50) !important; 
}

.bubble.bot-row { 
  background-color: var(--color-neutral-50) !important; 
}

/* ==========================================================================
   4. KOTAK PERCAKAPAN & KONTEKS RETRIEVAL (.soft-box)
   ========================================================================== */

.soft-box { 
  background-color: var(--block-background-fill) !important;
  border: 1px solid var(--border-color-primary) !important; 
  opacity: 1 !important;
}

.soft-box,
.soft-box .wrap,
.soft-box .wrap * {
  opacity: 1 !important;
}

/* Isi chat & konteks retrieval mengikuti tema: teks gelap, bubble terang */
.soft-box .prose,
.soft-box .prose *,
.soft-box .wrap span,
.soft-box .wrap p,
.soft-box span.text {
  color: var(--body-text-color) !important;
  -webkit-text-fill-color: var(--body-text-color) !important;
  background-color: transparent !important;
  font-weight: 400 !important;
  visibility: visible !important;
}

/* Kotak Konteks Retrieval: samakan dengan kolom Percakapan -
   latar biru gelap + teks putih keabuan terang, tanpa gradient penutup */
.soft-box textarea[data-testid="textbox"],
.soft-box textarea[data-testid="textbox"]:disabled,
.soft-box textarea[data-testid="textbox"][readonly],
.soft-box .wrap textarea {
  background-color: var(--primary-700) !important;
  color: #f1f5f9 !important;
  -webkit-text-fill-color: #f1f5f9 !important;
  caret-color: #f1f5f9 !important;
  border: none !important;
  box-shadow: none !important;
  opacity: 1 !important;
  resize: none !important;
}

/* ==========================================================================
   5. STYLING TABEL & DATAFRAME (Teks Pertanyaan Terlihat)
   ========================================================================== */

.table-wrap, 
table.dataframe { 
  border-radius: 0 !important; 
}

table.dataframe th, 
table.dataframe td { 
  border-radius: 0 !important; 
}

table.dataframe th { 
  white-space: nowrap !important; 
}

/* Perbaikan khusus agar teks di sel tabel pertanyaan terlihat jelas */
table.dataframe td { 
  white-space: normal !important; 
  word-break: break-word !important; 
  color: #f1f5f9 !important;
  -webkit-text-fill-color: #f1f5f9 !important;
}

/* Tabel rekap (Gradio 6) */
.recap-table .virtual-row .cell-wrap span,
.recap-table .header-cell .header-content span {
  white-space: normal !important;
  text-overflow: clip !important;
  word-break: break-word !important;
  overflow: visible !important;
  color: #f1f5f9 !important;
  -webkit-text-fill-color: #f1f5f9 !important;
}
"""

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=_CSS,
    )


if __name__ == "__main__":
    main()
