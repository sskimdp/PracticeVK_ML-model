// фоновый поток обработки изображений.
// асинхронный пайплайн: decoding → analyzing → applying → encoding,
// с прогрессом и отменой, чтобы не блокировать главный поток.
// поддерживает повторный пересчёт с ручными параметрами (reapply):
// декодированный оригинал кэшируется, чтобы не декодировать заново.

import type {
  CorrectionParams,
  RequestMessage,
  ResponseMessage,
  TaskId,
  TaskStatus,
  WorkerSource,
} from '../api/types.ts';
import { CancelledError, sleep } from './util.ts';
import { decodeSource, type DecodedImage } from './decode.ts';
import { applyCorrection } from './correction.ts';
import { MODEL_INPUT_SIZE, predictParams, ensureModel } from './model.ts';

const ctx = self as DedicatedWorkerGlobalScope;

/** максимальная сторона превью оригинала для UI */
const PREVIEW_MAX = 1100;

function post(msg: ResponseMessage, transfer?: Transferable[]): void {
  ctx.postMessage(msg, transfer ?? []);
}

interface TaskHandle {
  cancelled: boolean;
}
const tasks = new Map<TaskId, TaskHandle>();
/** кэш декодированных оригиналов для повторного пересчёта */
const origCache = new Map<TaskId, ImageData>();

/** url модели, присылается главным потоком сообщением init */
let modelUrl: string | null = null;

ctx.addEventListener('message', (e: MessageEvent<RequestMessage>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'init': {
      modelUrl = msg.modelUrl;
      void ensureModel(modelUrl).catch(() => {});
      break;
    }
    case 'create': {
      // активная задача одна — освобождаем кэш предыдущих
      origCache.clear();
      const handle: TaskHandle = { cancelled: false };
      tasks.set(msg.taskId, handle);
      void runTask(msg.taskId, msg.source, handle);
      break;
    }
    case 'cancel': {
      const handle = tasks.get(msg.taskId);
      if (handle) handle.cancelled = true;
      origCache.delete(msg.taskId);
      break;
    }
    case 'reapply': {
      void reapply(msg.reqId, msg.taskId, msg.params);
      break;
    }
  }
});

async function runTask(taskId: TaskId, source: WorkerSource, handle: TaskHandle): Promise<void> {
  const emit = (status: TaskStatus, progress: number, params?: CorrectionParams) =>
    post({ type: 'status', info: { taskId, status, progress, params } });

  const ensureLive = () => {
    if (handle.cancelled) throw new CancelledError();
  };

  try {
    emit('queued', 0);
    await sleep(0);

    // декодирование
    ensureLive();
    emit('decoding', 0);
    const decoded = await decodeSource(source, (p) => emit('decoding', p));

    // превью оригинала для UI + кэш полного оригинала для пересчёта
    const preview = await makePreview(decoded);
    const original = bitmapToImageData(decoded);
    decoded.bitmap.close();
    origCache.set(taskId, original);

    // анализ — подбор параметров моделью по уменьшенному thumbnail
    ensureLive();
    emit('analyzing', 0);
    const thumb = thumbnailFromImageData(original, MODEL_INPUT_SIZE);
    let params: CorrectionParams;
    try {
      if (!modelUrl) throw new Error('URL модели не задан');
      params = await predictParams(thumb, modelUrl);
    } catch {
      params = { brightness: 1, contrast: 1, saturation: 1 };
    }
    emit('analyzing', 1, params);

    // применяем коррекцию к копии, оригинал сохраняем в кэше
    ensureLive();
    emit('applying', 0, params);
    const working = cloneImageData(original);
    await applyCorrection(working, params, {
      onProgress: (p) => emit('applying', p, params),
      shouldCancel: () => handle.cancelled,
    });

    // кодирование результата
    ensureLive();
    emit('encoding', 0, params);
    const blob = await encodeImageData(working);
    emit('encoding', 1, params);

    ensureLive();
    emit('done', 1, params);
    post({ type: 'result', taskId, blob, preview, params });
  } catch (err) {
    origCache.delete(taskId);
    if (err instanceof CancelledError) {
      emit('cancelled', 0);
    } else {
      const message = err instanceof Error ? err.message : String(err);
      post({ type: 'error', taskId, error: message });
    }
  } finally {
    tasks.delete(taskId);
  }
}

/** повторное применение коррекции с ручными параметрами, полноразмерно */
async function reapply(reqId: string, taskId: TaskId, params: CorrectionParams): Promise<void> {
  const original = origCache.get(taskId);
  if (!original) {
    post({ type: 'error', taskId, error: 'Оригинал недоступен для пересчёта' });
    return;
  }
  const working = cloneImageData(original);
  await applyCorrection(working, params, { onProgress: () => {}, shouldCancel: () => false });
  const blob = await encodeImageData(working);
  post({ type: 'reapplyResult', reqId, blob });
}

function cloneImageData(src: ImageData): ImageData {
  return new ImageData(new Uint8ClampedArray(src.data), src.width, src.height);
}

/** уменьшенный thumbnail size×size для входа модели */
function thumbnailFromImageData(src: ImageData, size: number): ImageData {
  const srcCanvas = new OffscreenCanvas(src.width, src.height);
  srcCanvas.getContext('2d')!.putImageData(src, 0, 0);
  const dst = new OffscreenCanvas(size, size);
  const c = dst.getContext('2d', { willReadFrequently: true })!;
  c.drawImage(srcCanvas, 0, 0, src.width, src.height, 0, 0, size, size);
  return c.getImageData(0, 0, size, size);
}

/** превью оригинала в png, уменьшенное до PREVIEW_MAX */
async function makePreview(decoded: DecodedImage): Promise<Blob> {
  const { bitmap, width, height } = decoded;
  const scale = Math.min(1, PREVIEW_MAX / Math.max(width, height));
  const w = Math.max(1, Math.round(width * scale));
  const h = Math.max(1, Math.round(height * scale));
  const canvas = new OffscreenCanvas(w, h);
  const c = canvas.getContext('2d')!;
  c.drawImage(bitmap, 0, 0, width, height, 0, 0, w, h);
  return canvas.convertToBlob({ type: 'image/png' });
}

function bitmapToImageData(decoded: DecodedImage): ImageData {
  const { bitmap, width, height } = decoded;
  const canvas = new OffscreenCanvas(width, height);
  const c = canvas.getContext('2d', { willReadFrequently: true });
  if (!c) throw new Error('OffscreenCanvas 2D context недоступен');
  c.drawImage(bitmap, 0, 0);
  return c.getImageData(0, 0, width, height);
}

async function encodeImageData(image: ImageData): Promise<Blob> {
  const canvas = new OffscreenCanvas(image.width, image.height);
  const c = canvas.getContext('2d');
  if (!c) throw new Error('OffscreenCanvas 2D context недоступен');
  c.putImageData(image, 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}
