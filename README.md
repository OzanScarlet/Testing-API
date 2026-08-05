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
| `modules/questions_generator.py` | Generator pertanyaan (chunk → LLM → `{"questions": [...]}`) |
| `modules/staging_client.py` | Kirim 1 pertanyaan ke chatbot (interaktif, terminal) |
| `modules/phoenix_extractor.py` | POST + ambil jawaban dari trace Phoenix |
| `modules/judge.py` | POST → ambil trace → nilai 3 kriteria (interaktif, terminal) |
| `modules/gradio_app.py` | UI Gradio: tulis pertanyaan → POST → trace → judge |
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

Alur: POST pertanyaan → ambil `request_id` dari response → cari LLM span di Phoenix yang `attributes.metadata.request_id`-nya cocok → ambil jawaban dari `attributes.output.value` (polling hingga trace muncul, default 30 detik).

## Nilai jawaban dengan AI judge

```bash
python modules/judge.py
# lalu ketik pertanyaan di prompt
```

Alur: POST pertanyaan → ambil jawaban FastAPI + `request_id` → ambil jawaban acuan dari trace Phoenix → nilai 3 kriteria (akurasi, kelengkapan, kesesuaian konteks), skor 0–10 + label + kesimpulan.

## UI Gradio

```bash
python modules/gradio_app.py
# lalu buka http://127.0.0.1:7860
```

Tulis pertanyaan, klik **Evaluasi** — hasilnya menampilkan jawaban FastAPI, jawaban trace Phoenix, dan penilaian judge.

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
