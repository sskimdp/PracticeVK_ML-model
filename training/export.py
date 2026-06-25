"""экспорт обученной модели в формат TensorFlow.js

конвертирует Keras-модель (.h5) в tfjs_layers_model с квантованием
весов (uint16); это заметно уменьшает размер при минимальной потере
точности, что важно для запуска в браузере
"""

import os
import shutil
import subprocess
import sys

import config as cfg


def main() -> None:
    h5_path = cfg.ROOT / "model.h5"
    if not h5_path.exists():
        sys.exit(f"Не найдена модель {h5_path}. Сначала запустите train.py")

    out = cfg.TFJS_OUTPUT
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # tensorflowjs_converter внутри разбивает аргументы по пробелам, поэтому
    # пути с пробелами ломают его; запускаем из каталога training с
    # относительными путями без пробелов
    converter = os.path.join(".venv", "bin", "tensorflowjs_converter")
    rel_h5 = os.path.relpath(h5_path, cfg.ROOT)
    rel_out = os.path.relpath(out, cfg.ROOT)
    # флаг квантования ставим после позиционных путей, иначе его nargs='?'
    # съедает путь
    cmd = [
        converter,
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        rel_h5,
        rel_out,
        "--quantize_uint16=*",  # квантование весов до 16 бит
    ]
    print("Конвертация:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cfg.ROOT)
    # конвертер шумит предупреждениями про tfdf, фильтруем шум, ошибки оставляем
    for line in (result.stdout + result.stderr).splitlines():
        low = line.lower()
        if any(k in low for k in ("error", "traceback", "exception", "fail")):
            print(line)
    if result.returncode != 0:
        sys.exit(f"Конвертация завершилась с кодом {result.returncode}")

    total = sum(f.stat().st_size for f in out.glob("*"))
    print("\nФайлы модели TF.js:")
    for f in sorted(out.glob("*")):
        print(f"  {f.name}: {f.stat().st_size / 1024:.1f} КБ")
    print(f"Суммарный размер модели: {total / 1024:.1f} КБ")


if __name__ == "__main__":
    main()
