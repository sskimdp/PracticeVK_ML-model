// загрузка и инференс ml-модели в воркере.
// модель принимает thumbnail 128×128 и возвращает три множителя коррекции
// (яркость, контраст, насыщенность). tf.js грузится лениво, чтобы не утяжелять
// старт воркера. бэкенд выбирается с фолбэком webgl → cpu.

import type { CorrectionParams } from '../api/types.ts';
import { clamp } from './util.ts';

/** размер входа модели, должен совпадать с обучением */
export const MODEL_INPUT_SIZE = 128;

/** границы множителей, чтобы исключить экстремальную коррекцию */
const FACTOR_MIN = 0.4;
const FACTOR_MAX = 2.5;

type TF = typeof import('@tensorflow/tfjs');

let tfPromise: Promise<TF> | null = null;
let model: import('@tensorflow/tfjs').LayersModel | null = null;
let loadPromise: Promise<void> | null = null;
let backendName = '';

function getTf(): Promise<TF> {
  if (!tfPromise) tfPromise = import('@tensorflow/tfjs');
  return tfPromise;
}

async function selectBackend(tf: TF): Promise<void> {
  for (const backend of ['webgl', 'cpu']) {
    try {
      const ok = await tf.setBackend(backend);
      if (ok) {
        await tf.ready();
        backendName = backend;
        return;
      }
    } catch {
      // пробуем следующий
    }
  }
  // гарантированный фолбэк
  await tf.setBackend('cpu');
  await tf.ready();
  backendName = 'cpu';
}

/** загружает модель и прогревает её для стабильного времени инференса */
export async function ensureModel(modelUrl: string): Promise<void> {
  if (model) return;
  if (!loadPromise) {
    loadPromise = (async () => {
      const tf = await getTf();
      await selectBackend(tf);
      model = await tf.loadLayersModel(modelUrl);
      // первый прогон компилирует шейдеры и прогревает бэкенд
      const warm = tf.zeros([1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3]);
      const out = model.predict(warm) as import('@tensorflow/tfjs').Tensor;
      await out.data();
      warm.dispose();
      out.dispose();
    })().catch((err) => {
      // сбрасываем кэш, чтобы следующая попытка загрузила модель заново
      loadPromise = null;
      model = null;
      throw err;
    });
  }
  await loadPromise;
}

export function getBackendName(): string {
  return backendName;
}

/** предсказывает параметры коррекции по thumbnail 128×128 */
export async function predictParams(
  thumbnail: ImageData,
  modelUrl: string,
): Promise<CorrectionParams> {
  await ensureModel(modelUrl);
  const tf = await getTf();
  if (!model) throw new Error('Модель не загружена');

  const factors = tf.tidy(() => {
    const input = tf.browser
      .fromPixels(thumbnail)
      .toFloat()
      .div(255)
      .expandDims(0); // [1,128,128,3]
    const pred = model!.predict(input) as import('@tensorflow/tfjs').Tensor;
    return pred.dataSync();
  });

  return {
    brightness: clamp(factors[0], FACTOR_MIN, FACTOR_MAX),
    contrast: clamp(factors[1], FACTOR_MIN, FACTOR_MAX),
    saturation: clamp(factors[2], FACTOR_MIN, FACTOR_MAX),
  };
}

/** освобождение ресурсов модели */
export function disposeModel(): void {
  model?.dispose();
  model = null;
  loadPromise = null;
}
