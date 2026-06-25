// человекочитаемые подписи статусов для ui

import type { TaskStatus } from '../api/types.ts';

export const STATUS_LABELS: Record<TaskStatus, string> = {
  queued: 'В очереди',
  decoding: 'Декодирование',
  analyzing: 'Анализ изображения',
  applying: 'Применение коррекции',
  encoding: 'Сохранение результата',
  done: 'Готово',
  error: 'Ошибка',
  cancelled: 'Отменено',
};

/** форматирование размера файла */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} КБ`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;
}

/** форматирование разрешения в мегапикселях */
export function formatMegapixels(width: number, height: number): string {
  const mp = (width * height) / 1_000_000;
  return `${width}×${height} (${mp.toFixed(1)} Мпк)`;
}
