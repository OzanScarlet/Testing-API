import os

import requests
from dotenv import load_dotenv

load_dotenv()

CHATOPA_URL = os.getenv("CHATOPA_URL")
CHATOPA_API_KEY = os.getenv("CHATOPA_API_KEY")


def main():
    question = input("Pertanyaan: ").strip()
    if not question:
        print("Pertanyaan kosong.")
        return

    headers = {"x-api-key": CHATOPA_API_KEY}
    data = {"content": question}

    print(f"POST {CHATOPA_URL}")
    print(f"Body (form-data): content = '{question}'")
    try:
        r = requests.post(CHATOPA_URL, headers=headers, data=data, timeout=180)
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:500]}")
    except Exception as e:
        print(f"Gagal terhubung ke {CHATOPA_URL}: {e}")


if __name__ == "__main__":
    main()