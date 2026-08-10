# Testing-API — Evaluasi Chatbot RAG Kelapa Sawit

Membaca semua dokumen `*.md` di dalam folder (rekursif), lalu LLM generate daftar pertanyaan uji per chunk, disimpan ke `output/questions.json`. Pertanyaan-pertanyaan itu lalu dikirim ke chatbot (FastAPI staging), jawabannya diambil dari trace Phoenix, lalu dinilai AI judge (3 kriteria) — bisa interaktif lewat terminal atau lewat UI Gradio.

## Cara menjalankan

```bash
pip install -r requirements.txt
cp .env.example .env   # lalu isi API key
python modules/questions_generator.py            # generate pertanyaan (folder data/)
python modules/questions_generator.py data/ksi   # proses folder tertentu
```

Output: `output/questions.json`. Jalankan ulang untuk **resume otomatis** — file/chunk yang sudah selesai dilewati.

## Struktur

| File | Fungsi |
|------|--------|
| `modules/questions_generator.py` | Generator pertanyaan (chunk → LLM → `{"questions": [...]}`). Untuk folder/CLI (`.md`) atau upload multi-file (`.md/.txt/.pdf/.docx/.doc` via `generate_questions_from_files`) |
| `modules/staging_client.py` | Kirim 1 pertanyaan ke chatbot (interaktif, terminal) |
| `modules/phoenix_extractor.py` | POST + ambil jawaban / konteks retrieval dari trace Phoenix (span `retrieve_context_and_extract`) |
| `modules/judge.py` | POST → ambil trace → nilai 3 kriteria (interaktif, terminal) |
| `modules/gradio_app.py` | Dashboard Gradio: 2 tab (Evaluasi + Rekapitulasi). Evaluasi: pilih dokumen → pertanyaan auto sequence 1 per 1 (atau ketik manual) → POST → trace → judge (tabel + grafik), hasil disimpan ke `output/evaluations.jsonl`. Bisa **Auto Evaluasi** (semua file berurutan tanpa klik, pindah file otomatis, retry sampai sukses) atau **1 Tombol Generate → Evaluasi** (upload banyak file `.md/.txt/.pdf/.docx/.doc` → generate pertanyaan → langsung POST → trace → judge). Rekapitulasi: tabel hasil dari jsonl |
| `data/` | Kumpulan dokumen `.md` sumber |
| `output/questions.json` | Hasil generate |

## Kirim pertanyaan ke chatbot

```bash
python modules/staging_client.py
# lalu ketik pertanyaan di prompt
```

Script mengirim POST `form-data` (key `content` = pertanyaan) dengan header `x-api-key` ke chatbot. URL & key dibaca dari `.env` (`CHATOPA_URL`, `CHATOPA_API_KEY`).

## Ambil jawaban dari trace Phoenix

```bash
python modules/phoenix_extractor.py
# lalu ketik pertanyaan di prompt
```

Alur: POST pertanyaan → ambil `request_id` dari response → cari trace di Phoenix yang `attributes.metadata.request_id`-nya cocok (polling hingga trace muncul, default 60 detik) → ambil jawaban dari `attributes.output.value`, atau konteks retrieval dari span `retrieve_context_and_extract` (`output.value.retrieval.context`). Konteks retrieval dibaca dari span ini sesuai arahan pembimbing (bukan `_choose_agent`).

## Nilai jawaban dengan AI judge

```bash
python modules/judge.py
# lalu ketik pertanyaan di prompt
```

Alur: POST pertanyaan → ambil jawaban FastAPI + `request_id` → ambil **konteks retrieval** dari trace Phoenix (dokumen hasil retrieval, bukan jawaban trace) → nilai 3 kriteria (akurasi, kelengkapan, kesesuaian konteks), skor 0–10 + label + kesimpulan.

> Catatan: yang dinilai adalah **jawaban FastAPI** (yang dilihat user). Acuan penilaiannya **konteks retrieval** dari trace Phoenix — karena trace merekam request yang sama persis dengan yang dikirim ke FastAPI, membandingkan jawaban FastAPI vs jawaban trace tidak bermakna (hasilnya selalu sama).

## UI Gradio

```bash
python modules/gradio_app.py
# lalu buka http://127.0.0.1:7860
```

Dashboard 2 tab:

**Tab Evaluasi** — pilih **dokumen** (dari `output/questions.json`) → pertanyaan **diload otomatis satu per satu** (auto sequence 1 per 1), melewati yang sudah dinilai. Bisa juga **ketik manual**. Tombol:

- **Evaluasi** — menilai satu pertanyaan (pertanyaan aktif dari dokumen, atau manual). Semua hasil sebelumnya langsung dibersihkan saat tombol ditekan.
- **Auto Evaluasi** — menilai **semua dokumen berurutan tanpa klik**. Untuk tiap pertanyaan: pipeline POST → trace → judge dijalankan dengan **retry** sampai semua step keluar hasil; baru setelah soal berhasil lanjut ke soal berikutnya. Setelah seluruh soal satu dokumen tuntas, otomatis pindah ke dokumen berikutnya (dropdown ikut berpindah). Jika suatu soal gagal dinilai setelah 10 percobaan, proses **dihentikan** (soal tidak dianggap selesai) agar tidak ada step/pertanyaan yang tertinggal — periksa tracing/retrieval, lalu klik Auto Evaluasi lagi.
- **1 Tombol: Generate → Evaluasi** — **upload satu atau beberapa file** (`.md`, `.txt`, `.pdf`, `.docx`, `.doc`), satu klik menjalankan seluruh alur: **generate pertanyaan** dari file-file tersebut → langsung **POST** ke chatbot → bandingkan dengan **trace Phoenix** → nilai **judge**, untuk semua pertanyaan secara otomatis. Hasil generate tersimpan ke `output/questions_upload.json`.
- **Stop** — menghentikan Auto Evaluasi / Generate→Evaluasi.

Bagian tampilan:
- **Tengah**: jawaban FastAPI (Chatbot) di kiri; **konteks retrieval** dari trace Phoenix di kanan.
- **Bawah**: **tabel skor judge** (Akurasi / Kelengkapan / Kesesuaian / Total) + **grafik batang** per kriteria + ringkasan.
- Setelah evaluasi selesai, pertanyaan otomatis maju ke pertanyaan berikutnya yang belum dinilai.
- Setiap hasil evaluasi disimpan ke `output/evaluations.jsonl` (satu baris JSON per evaluasi: timestamp, pertanyaan, request_id, jawaban, preview konteks, skor 3 kriteria, total, label, kesimpulan, latency).

**Tab Rekapitulasi** — tabel semua hasil evaluasi dari `output/evaluations.jsonl` (Timestamp, Pertanyaan, Total, Label, Request ID); klik **Muat Ulang** untuk refresh.

Konfigurasi dibaca dari `.env`:
- `PHOENIX_BASE_URL` — base URL Phoenix (contoh `https://phoenix.example.com`)
- `PHOENIX_PROJECT_NAME` — nama project tempat trace chatbot masuk
- `PHOENIX_API_KEY` — token akses Phoenix (client menambah header `Authorization: Bearer` sendiri, jadi isi **tanpa** awalan `Bearer `)

## Konfigurasi (`.env`)

**LLM (OpenAI / relay OpenAI-compatible):**
- Relay (aktif saat ini): `BASE_URL=https://ai.api-relay.my.id/v1`, `MODEL_NAME=deepseek-v4-flash`, `OPENAI_API_KEY=...`
- OpenAI resmi: `BASE_URL=https://api.openai.com/v1`, `MODEL_NAME=gpt-4o-mini`, `OPENAI_API_KEY=...`

**Anti-limit:** `CHUNK_CHARS` (ukuran chunk, default 5000), `MAX_QUESTIONS_PER_CHUNK` (default 10), `DELAY` (jeda antar request, default 3 detik).

**Chatbot:** `CHATOPA_URL` (endpoint `/chat`), `CHATOPA_API_KEY` (header `x-api-key`).

**Phoenix:** `PHOENIX_BASE_URL`, `PHOENIX_PROJECT_NAME`, `PHOENIX_API_KEY` (tanpa awalan `Bearer `).

## Catatan

- Generator butuh `response_format: json_object` — didukung OpenAI dan relay OpenAI-compatible.
- Beberapa relay/proxy API memblokir header `User-Agent` khas library OpenAI SDK (403 "Your request was blocked"). Sudah diatasi otomatis dengan menimpa UA di `get_client()`.
