// декодирование изображений в воркере.
// jpg/png/bmp — через нативный createImageBitmap, heic уже конвертирован
// в png в главном потоке (heic2any требует window).

import type { WorkerSource } from '../api/types.ts';

/** ограничение разрешения */
export const MAX_MEGAPIXELS = 15;
const MAX_PIXELS = MAX_MEGAPIXELS * 1_000_000;

export interface DecodedImage {
  bitmap: ImageBitmap;
  width: number;
  height: number;
}

/**
 * декодирует источник в ImageBitmap и проверяет ограничение по разрешению.
 * onProgress: 0..1 в пределах стадии декодирования.
 */
export async function decodeSource(
  source: WorkerSource,
  onProgress: (p: number) => void,
): Promise<DecodedImage> {
  onProgress(0.05);

  let bitmap: ImageBitmap;

  if (source.kind === 'imagedata') {
    const data = new Uint8ClampedArray(source.buffer);
    const imageData = new ImageData(data, source.width, source.height);
    bitmap = await createImageBitmap(imageData);
  } else {
    // heic уже конвертирован в png в главном потоке, здесь только нативный декод
    const blob = new Blob([source.buffer], {
      type: source.mime || 'application/octet-stream',
    });
    bitmap = await createImageBitmap(blob).catch((err) => {
      throw new Error(
        `Не удалось декодировать изображение. Формат может не поддерживаться браузером. (${
          err instanceof Error ? err.message : String(err)
        })`,
      );
    });
  }

  onProgress(0.9);

  const { width, height } = bitmap;
  if (width * height > MAX_PIXELS) {
    bitmap.close();
    const mp = ((width * height) / 1_000_000).toFixed(1);
    throw new Error(
      `Изображение слишком большое: ${width}×${height} (${mp} Мпк). Максимум – ${MAX_MEGAPIXELS} Мпк.`,
    );
  }

  onProgress(1);
  return { bitmap, width, height };
}

