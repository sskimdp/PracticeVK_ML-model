// алгоритм коррекции изображения.
// применяет коэффициенты яркости, контраста и насыщенности, подобранные
// моделью. обработка идёт чанками с прогрессом и проверкой отмены, чтобы
// не блокировать воркер на больших изображениях.

import type { CorrectionParams } from '../api/types.ts';
import { CancelledError, clamp } from './util.ts';

// коэффициенты luma (rec. 601)
const LUMA_R = 0.299;
const LUMA_G = 0.587;
const LUMA_B = 0.114;

/** сколько пикселей обрабатываем между уступками управления */
const CHUNK_PIXELS = 500_000;

export interface ApplyOptions {
  onProgress: (p: number) => void;
  shouldCancel: () => boolean;
}

/**
 * применяет коррекцию к ImageData на месте.
 * порядок: яркость → контраст (через общий lut на 256 значений) → насыщенность
 * (зависит от всех каналов, считается попиксельно).
 */
export async function applyCorrection(
  image: ImageData,
  params: CorrectionParams,
  opts: ApplyOptions,
): Promise<void> {
  const { brightness, contrast, saturation } = params;
  const data = image.data;
  const pixelCount = image.width * image.height;

  // контраст считаем вокруг средней яркости, а не вокруг 128 — иначе
  // усиление контраста гасит прирост яркости на тёмных кадрах
  const mean = meanLuma(data, pixelCount);

  // яркость и контраст независимы по каналам, поэтому считаем результат
  // для каждого из 256 уровней один раз через lut
  const lut = buildBrightnessContrastLUT(brightness, contrast, mean);
  const needSaturation = Math.abs(saturation - 1) > 1e-3;

  let processed = 0;
  while (processed < pixelCount) {
    if (opts.shouldCancel()) throw new CancelledError();

    const end = Math.min(processed + CHUNK_PIXELS, pixelCount);
    for (let i = processed; i < end; i++) {
      const o = i * 4;
      let r = lut[data[o]];
      let g = lut[data[o + 1]];
      let b = lut[data[o + 2]];

      if (needSaturation) {
        const y = LUMA_R * r + LUMA_G * g + LUMA_B * b;
        r = clamp(y + (r - y) * saturation, 0, 255);
        g = clamp(y + (g - y) * saturation, 0, 255);
        b = clamp(y + (b - y) * saturation, 0, 255);
      }

      data[o] = r;
      data[o + 1] = g;
      data[o + 2] = b;
      // альфа не трогаем
    }

    processed = end;
    opts.onProgress(processed / pixelCount);
    // уступаем управление, чтобы воркер мог принять сообщение об отмене
    await Promise.resolve();
  }
}

/** средняя яркость luma изображения в диапазоне 0..255 */
function meanLuma(data: Uint8ClampedArray, pixelCount: number): number {
  // сэмплируем каждый 16-й пиксель — быстро и достаточно точно для оценки среднего
  let sum = 0;
  let count = 0;
  const step = pixelCount > 1_000_000 ? 16 : 1;
  for (let i = 0; i < pixelCount; i += step) {
    const o = i * 4;
    sum += LUMA_R * data[o] + LUMA_G * data[o + 1] + LUMA_B * data[o + 2];
    count++;
  }
  return count ? sum / count : 128;
}

/**
 * строит таблицу на 256 значений: вход — уровень канала 0..255,
 * выход — после контраста (вокруг mean) и яркости, с клиппингом.
 */
function buildBrightnessContrastLUT(
  brightness: number,
  contrast: number,
  mean: number,
): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256);
  for (let v = 0; v < 256; v++) {
    // контраст вокруг средней яркости, затем яркость
    const x = brightness * ((v - mean) * contrast + mean);
    lut[v] = clamp(x, 0, 255);
  }
  return lut;
}
