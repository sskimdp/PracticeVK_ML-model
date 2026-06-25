"""подготовка обучающих данных: self-supervised авто-улучшение

идея:
  1. берём набор реальных фотографий
  2. создаём разнообразие входов: часть кадров случайно искажаем по яркости,
     контрасту и насыщенности, часть оставляем как есть
  3. для каждого входа аналитически вычисляем идеальные факторы коррекции,
     приводящие его к эталонной статистике (авто-уровни по гистограмме и
     цель по насыщенности), см. ideal_factors()
  4. модель учится предсказывать эти факторы

так нормальные фото получают аккуратное улучшение, а плохие — сильное;
разметка не нужна, целевые значения вычисляются точно
"""

import io
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import config as cfg


def _download_one(index: int) -> bool:
    """скачивает одно фото с детерминированным seed, возвращает успех"""
    path = cfg.RAW_DIR / f"{index:04d}.jpg"
    if path.exists() and path.stat().st_size > 0:
        return True
    url = f"https://picsum.photos/seed/{index}/{cfg.BASE_IMG_SIZE}/{cfg.BASE_IMG_SIZE}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        # проверяем, что это валидное изображение
        Image.open(io.BytesIO(data)).convert("RGB").verify()
        path.write_bytes(data)
        return True
    except Exception:
        return False


def ensure_base_images() -> None:
    """скачивает недостающие базовые изображения параллельно"""
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(cfg.RAW_DIR.glob("*.jpg")))
    if existing >= cfg.NUM_BASE_IMAGES:
        print(f"Базовые изображения уже на месте: {existing}")
        return
    print(f"Скачиваю базовые изображения (нужно {cfg.NUM_BASE_IMAGES})…")
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(_download_one, range(cfg.NUM_BASE_IMAGES)))
    ok = sum(results)
    print(f"Готово: {ok}/{cfg.NUM_BASE_IMAGES} изображений")
    if ok < 20:
        raise RuntimeError(
            "Не удалось скачать достаточно изображений. Проверьте интернет "
            "или положите свои фото в training/data/raw/*.jpg"
        )


def load_base_arrays() -> np.ndarray:
    """загружает базовые изображения в массив uint8 [N, S, S, 3]"""
    files = sorted(cfg.RAW_DIR.glob("*.jpg"))
    if not files:
        raise RuntimeError("Нет базовых изображений — запустите ensure_base_images().")
    arrays = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB").resize((cfg.BASE_IMG_SIZE, cfg.BASE_IMG_SIZE))
            arrays.append(np.asarray(img, dtype=np.uint8))
        except Exception:
            continue
    print(f"Загружено базовых изображений: {len(arrays)}")
    return np.stack(arrays, axis=0)


def ideal_factors(img: np.ndarray) -> tuple:
    """аналитически вычисляет идеальные факторы коррекции для изображения

    тон-коррекция в нашей формуле — аффинное преобразование яркости
    out = A*v + B (A = b*c, контраст вокруг среднего); поэтому по гистограмме
    можно посчитать авто-уровни и разложить их обратно в (b, c); насыщенность —
    подтяжка средней цветности к цели; всё с мягкой силой и клиппингом
    """
    lr, lg, lb = cfg.LUMA
    luma = img[..., 0] * lr + img[..., 1] * lg + img[..., 2] * lb
    mu = float(luma.mean())
    p_low = float(np.percentile(luma, cfg.LEVELS_LOW_PCT))
    p_high = float(np.percentile(luma, cfg.LEVELS_HIGH_PCT))
    span = max(p_high - p_low, 0.04)

    # полная авто-растяжка: [p_low, p_high] -> [target_low, target_high]
    a_full = (cfg.LEVELS_TARGET_HIGH - cfg.LEVELS_TARGET_LOW) / span
    b_full = cfg.LEVELS_TARGET_LOW - a_full * p_low
    # смешиваем с тождеством для мягкой силы
    a = cfg.LEVELS_STRENGTH
    A = (1 - a) + a * a_full
    B = a * b_full

    # раскладываем аффинное (out = A*v + B) в наши (b, c) вокруг среднего mu:
    #   A = b*c,  B = b*mu*(1-c) = b*mu - A*mu  =>  b = A + B/mu,  c = A/b
    mu_safe = max(mu, 1e-3)
    bright = float(np.clip(A + B / mu_safe, cfg.FACTOR_MIN, cfg.FACTOR_MAX))
    contrast = float(np.clip(A / max(bright, 1e-3), cfg.FACTOR_MIN, cfg.FACTOR_MAX))

    # среднюю HSV-насыщенность (яркостно-инвариантную) тянем к цели
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    chroma = float(((mx - mn) / (mx + 1e-3)).mean())
    s_full = cfg.TARGET_CHROMA / max(chroma, 1e-3)
    sa = cfg.SATURATION_STRENGTH
    sat = float(np.clip((1 - sa) + sa * s_full, cfg.FACTOR_MIN, cfg.FACTOR_MAX))

    return bright, contrast, sat


def apply_adjust(img: np.ndarray, b: float, c: float, s: float) -> np.ndarray:
    """применяет коррекцию к img: float32 [H,W,3] в диапазоне [0,1]

    контраст считается вокруг средней яркости изображения (как в correction.ts),
    что сохраняет прирост яркости на тёмных кадрах
    """
    lr, lg, lb = cfg.LUMA
    mean = float((img[..., 0] * lr + img[..., 1] * lg + img[..., 2] * lb).mean())
    out = b * ((img - mean) * c + mean)  # контраст вокруг среднего и яркость
    out = np.clip(out, 0.0, 1.0)
    luma = out[..., 0:1] * lr + out[..., 1:2] * lg + out[..., 2:3] * lb
    out = luma + (out - luma) * s  # насыщенность
    return np.clip(out, 0.0, 1.0)


class PairSequence:
    """генератор пар: искажённое изображение и факторы коррекции для него"""

    def __init__(self, base: np.ndarray, batch_size: int, steps: int, seed: int = 0):
        self.base = base
        self.batch_size = batch_size
        self.steps = steps
        self.rng = np.random.default_rng(seed)
        self.size = cfg.IMG_SIZE

    def _rand_crop(self, img_u8: np.ndarray) -> np.ndarray:
        s = cfg.BASE_IMG_SIZE
        d = self.size
        if s == d:
            crop = img_u8
        else:
            y = int(self.rng.integers(0, s - d + 1))
            x = int(self.rng.integers(0, s - d + 1))
            crop = img_u8[y : y + d, x : x + d]
        if self.rng.random() < 0.5:  # горизонтальный флип
            crop = crop[:, ::-1]
        return crop.astype(np.float32) / 255.0

    def batch(self):
        n = self.base.shape[0]
        X = np.empty((self.batch_size, self.size, self.size, 3), dtype=np.float32)
        Y = np.empty((self.batch_size, cfg.OUTPUTS), dtype=np.float32)
        for i in range(self.batch_size):
            base_img = self._rand_crop(self.base[int(self.rng.integers(0, n))])
            # создаём разнообразие входов: с вероятностью 0.8 искажаем кадр
            # (тёмный, яркий, блёклый), остальное оставляем как есть; цель не
            # зависит от искажения, это идеальные факторы для данного входа
            if self.rng.random() < 0.8:
                axes = [0, 1, 2]
                k = int(self.rng.integers(1, 4))
                active = self.rng.choice(axes, size=k, replace=False)
                bd = float(self.rng.uniform(*cfg.BRIGHTNESS_RANGE)) if 0 in active else 1.0
                cd = float(self.rng.uniform(*cfg.CONTRAST_RANGE)) if 1 in active else 1.0
                sd = float(self.rng.uniform(*cfg.SATURATION_RANGE)) if 2 in active else 1.0
                inp = apply_adjust(base_img, bd, cd, sd)
            else:
                inp = base_img
            X[i] = inp
            # цель — идеальные факторы, приводящие вход к эталонной статистике
            Y[i] = ideal_factors(inp)
        return X, Y

    def tf_dataset(self):
        import tensorflow as tf

        def gen():
            for _ in range(self.steps):
                yield self.batch()

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec((None, self.size, self.size, 3), tf.float32),
                tf.TensorSpec((None, cfg.OUTPUTS), tf.float32),
            ),
        ).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    ensure_base_images()
    base = load_base_arrays()
    seq = PairSequence(base, batch_size=4, steps=1)
    X, Y = seq.batch()
    print("X", X.shape, X.dtype, "min/max", float(X.min()), float(X.max()))
    print("Y", Y.shape, Y.dtype, "\n", Y)
