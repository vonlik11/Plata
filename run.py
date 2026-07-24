"""
Скрипт запуска: генерация датасета + запуск дашборда.
"""

import subprocess
import sys
import os


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, "data", "logs.csv")

    # 1. Генерируем датасет если его нет
    if not os.path.exists(data_path):
        print("Генерирую демо-датасет...")
        subprocess.run([sys.executable, os.path.join(base, "generate_dataset.py")], check=True)
    else:
        print(f"Демо-датасет уже существует: {data_path}")

    # 2. Запускаем Streamlit
    print("Запускаю дашборд...")
    print("Откройте http://localhost:8501 в браузере")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(base, "prompt_radar", "dashboard.py"),
        "--server.headless", "true",
    ], check=True)


if __name__ == "__main__":
    main()
