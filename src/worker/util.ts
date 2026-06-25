// служебные утилиты воркера.

/** ошибка прерывания задачи, отличает отмену от настоящих ошибок */
export class CancelledError extends Error {
  constructor() {
    super('cancelled');
    this.name = 'CancelledError';
  }
}

/** асинхронная пауза, в том числе для уступки управления */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** ограничение значения диапазоном [min, max] */
export function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}
