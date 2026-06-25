"""архитектура модели подбора параметров коррекции

задача — регрессия: изображение 128x128x3 в 3 множителя (яркость,
контраст, насыщенность); сеть компактная: для подбора глобальных
характеристик тона и цвета не нужна большая ёмкость, а малый размер
важен для запуска в браузере
"""

import tensorflow as tf

import config as cfg


def build_model() -> tf.keras.Model:
    L = tf.keras.layers
    inp = L.Input((cfg.IMG_SIZE, cfg.IMG_SIZE, 3))

    def block(x, filters):
        x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.MaxPooling2D()(x)
        return x

    x = block(inp, 16)   # 64x64
    x = block(x, 32)     # 32x32
    x = block(x, 48)     # 16x16
    x = block(x, 64)     # 8x8
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dense(32, activation="relu")(x)
    # выход — положительные множители около 1.0; softplus гарантирует
    # положительность и устойчивость обучения
    out = L.Dense(cfg.OUTPUTS, activation="softplus")(x)

    return tf.keras.Model(inp, out, name="correction_regressor")


if __name__ == "__main__":
    m = build_model()
    m.summary()
    params = m.count_params()
    print(f"\nПараметров: {params:,}  (~{params * 4 / 1024:.0f} КБ float32)")
