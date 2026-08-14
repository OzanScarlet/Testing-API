"""Kontrol global tracing OpenAI ke Phoenix.

Tracing terpasang sekali untuk seluruh proses app (OpenAIInstrumentor global
+ phoenix.otel.register). Modul ini menyediakan saklar agar tracing bisa
dimatikan dari UI tanpa mengubah behavior evaluasi itu sendiri.
"""
import threading

_disabled = False
_lock = threading.Lock()


def is_disabled() -> bool:
    return _disabled


def disable() -> str:
    """Matikan tracing global: uninstrument OpenAI + shutdown tracer."""
    global _disabled
    with _lock:
        if _disabled:
            return "Tracing sudah nonaktif."
        _disabled = True
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().uninstrument()
    except Exception as e:
        print(f"[TRACE] Gagal uninstrument OpenAI: {e}")
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if provider is not None:
            provider.shutdown()
    except Exception as e:
        print(f"[TRACE] Gagal shutdown tracer: {e}")
    print("[TRACE] Tracing dinonaktifkan.")
    return "Tracing **nonaktif** — tidak ada LLM call yang dikirim ke Phoenix."


def enable() -> str:
    """Aktifkan kembali tracing (re-instrument dilakukan lazy saat dipakai)."""
    global _disabled
    with _lock:
        _disabled = False
    print("[TRACE] Tracing diaktifkan kembali.")
    return "Tracing **aktif** — LLM call terkirim ke Phoenix."


def status() -> str:
    if _disabled:
        return "Tracing **nonaktif** — tidak ada LLM call yang dikirim ke Phoenix."
    return "Tracing **aktif** — LLM call terkirim ke Phoenix."
