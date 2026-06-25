"""обучение модели подбора параметров коррекции"""

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

import config as cfg
from dataset import PairSequence, ensure_base_images, load_base_arrays
from model import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--val-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    ensure_base_images()
    base = load_base_arrays()

    # делим базовые изображения на train и val, чтобы валидация шла по
    # неизвестным модели картинкам
    n = base.shape[0]
    idx = np.random.default_rng(args.seed).permutation(n)
    cut = max(1, int(n * 0.85))
    train_base, val_base = base[idx[:cut]], base[idx[cut:]]
    print(f"train/val базовых: {train_base.shape[0]}/{val_base.shape[0]}")

    train_ds = PairSequence(train_base, args.batch, args.steps, seed=args.seed).tf_dataset()
    val_ds = PairSequence(val_base, args.batch, args.val_steps, seed=args.seed + 1).tf_dataset()

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(1e-3),  # быстрее на apple silicon
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # сохраняем в обоих форматах: .keras (актуальный) и .h5 (для конвертера TF.js)
    model.save(cfg.MODEL_KERAS)
    h5_path = cfg.ROOT / "model.h5"
    model.save(h5_path)
    print(f"\nМодель сохранена:\n  {cfg.MODEL_KERAS}\n  {h5_path}")

    _sanity_check(model, val_base, args.seed)


def _sanity_check(model: tf.keras.Model, val_base: np.ndarray, seed: int) -> None:
    """краткая проверка: насколько предсказанные факторы близки к идеальным"""
    from dataset import apply_adjust, ideal_factors

    rng = np.random.default_rng(seed + 99)
    size = cfg.IMG_SIZE
    N = min(64, val_base.shape[0])
    X = np.empty((N, size, size, 3), np.float32)
    Y = np.empty((N, 3), np.float32)
    for i in range(N):
        img = val_base[i, :size, :size].astype(np.float32) / 255.0
        bd = float(rng.uniform(*cfg.BRIGHTNESS_RANGE))
        cd = float(rng.uniform(*cfg.CONTRAST_RANGE))
        sd = float(rng.uniform(*cfg.SATURATION_RANGE))
        X[i] = apply_adjust(img, bd, cd, sd)
        Y[i] = ideal_factors(X[i])
    pred = model.predict(X, verbose=0)
    mae = np.mean(np.abs(pred - Y), axis=0)
    print("\nПроверка на валидации (MAE по факторам):")
    print(f"  яркость:     {mae[0]:.3f}")
    print(f"  контраст:    {mae[1]:.3f}")
    print(f"  насыщенность:{mae[2]:.3f}")
    print(f"  средний MAE: {mae.mean():.3f}")


if __name__ == "__main__":
    main()
