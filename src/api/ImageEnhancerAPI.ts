// фасад системы в главном потоке: скрывает web worker и предоставляет
// методы createTask / getStatus / cancelTask / getResult и событие statusChange

import { TypedEmitter } from './emitter.ts';
import { isHeic, convertHeicToPng } from './heic.ts';
import {
  type ApiEventMap,
  type CorrectionParams,
  type RequestMessage,
  type ResponseMessage,
  type TaskId,
  type TaskStatusInfo,
  type WorkerSource,
  TERMINAL_STATUSES,
} from './types.ts';

/** входные форматы изображения, принимаемые методом createTask */
export type ImageInput = Blob | File | ImageData;

interface PendingResult {
  resolve: (blob: Blob) => void;
  reject: (err: Error) => void;
}

export class ImageEnhancerAPI extends TypedEmitter<ApiEventMap> {
  private worker: Worker;
  /** текущее состояние каждой задачи, обновляется из событий воркера */
  private states = new Map<TaskId, TaskStatusInfo>();
  /** готовые изображения */
  private results = new Map<TaskId, Blob>();
  /** превью оригиналов, декодированных воркером, для ui */
  private previews = new Map<TaskId, Blob>();
  /** ожидающие getResult, пока задача не завершится */
  private pendingResults = new Map<TaskId, PendingResult[]>();
  /** ожидающие reapply по reqId */
  private pendingReapply = new Map<string, PendingResult>();

  constructor() {
    super();
    // vite сам соберёт воркер и подставит правильный url
    this.worker = new Worker(new URL('../worker/worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker.addEventListener('message', (e: MessageEvent<ResponseMessage>) =>
      this.handleMessage(e.data),
    );

    // url модели вычисляем в главном потоке по document.baseURI — корректно
    // и в dev, и на хостинге
    const modelUrl = new URL('model/model.json', document.baseURI).href;
    this.worker.postMessage({ type: 'init', modelUrl } satisfies RequestMessage);
  }

  /** ставит задачу на обработку, возвращает её id */
  async createTask(image: ImageInput): Promise<TaskId> {
    // активная задача одна — освобождаем данные предыдущих ради экономии памяти
    this.states.clear();
    this.results.clear();
    this.previews.clear();

    const taskId = this.generateId();
    const source = await this.toWorkerSource(image);

    // сразу регистрируем начальное состояние, чтобы getStatus работал
    // даже до первого сообщения от воркера
    this.setState({ taskId, status: 'queued', progress: 0 });

    const message: RequestMessage = { type: 'create', taskId, source };
    this.worker.postMessage(message, [source.buffer]);
    return taskId;
  }

  /** текущий статус и прогресс задачи */
  async getStatus(taskId: TaskId): Promise<TaskStatusInfo> {
    const state = this.states.get(taskId);
    if (!state) throw new Error(`Задача не найдена: ${taskId}`);
    return { ...state };
  }

  /** прерывает задачу, возвращает признак успеха */
  async cancelTask(taskId: TaskId): Promise<{ success: boolean }> {
    const state = this.states.get(taskId);
    if (!state) return { success: false };
    if (TERMINAL_STATUSES.has(state.status)) {
      // уже завершена — отменять нечего
      return { success: false };
    }
    const message: RequestMessage = { type: 'cancel', taskId };
    this.worker.postMessage(message);
    return { success: true };
  }

  /** готовое изображение, ждёт завершения, если задача ещё в процессе */
  async getResult(taskId: TaskId): Promise<Blob> {
    const ready = this.results.get(taskId);
    if (ready) return ready;

    const state = this.states.get(taskId);
    if (!state) throw new Error(`Задача не найдена: ${taskId}`);
    if (state.status === 'error') throw new Error(state.error ?? 'Ошибка обработки');
    if (state.status === 'cancelled') throw new Error('Задача отменена');

    // задача ещё выполняется — ждём результат
    return new Promise<Blob>((resolve, reject) => {
      const list = this.pendingResults.get(taskId) ?? [];
      list.push({ resolve, reject });
      this.pendingResults.set(taskId, list);
    });
  }

  /** последние подобранные параметры коррекции, если есть */
  getParams(taskId: TaskId): CorrectionParams | undefined {
    return this.states.get(taskId)?.params;
  }

  /** превью оригинала, декодированное воркером, для отображения и ручного превью */
  getPreview(taskId: TaskId): Blob | undefined {
    return this.previews.get(taskId);
  }

  /**
   * повторно применяет коррекцию с ручными параметрами к полному изображению
   * для тонкой настройки, возвращает готовый blob
   */
  reapplyTask(taskId: TaskId, params: CorrectionParams): Promise<Blob> {
    const reqId = this.generateId();
    const message: RequestMessage = { type: 'reapply', reqId, taskId, params };
    this.worker.postMessage(message);
    return new Promise<Blob>((resolve, reject) => {
      this.pendingReapply.set(reqId, { resolve, reject });
    });
  }

  /** освобождает ресурсы */
  dispose(): void {
    this.worker.terminate();
    this.states.clear();
    this.results.clear();
    this.previews.clear();
    this.pendingResults.clear();
    this.pendingReapply.clear();
  }

  private handleMessage(msg: ResponseMessage): void {
    switch (msg.type) {
      case 'status':
        this.setState(msg.info);
        break;
      case 'result':
        this.results.set(msg.taskId, msg.blob);
        this.previews.set(msg.taskId, msg.preview);
        this.resolvePending(msg.taskId, msg.blob);
        break;
      case 'reapplyResult': {
        const pending = this.pendingReapply.get(msg.reqId);
        if (pending) {
          pending.resolve(msg.blob);
          this.pendingReapply.delete(msg.reqId);
        }
        break;
      }
      case 'error': {
        const prev = this.states.get(msg.taskId);
        this.setState({
          taskId: msg.taskId,
          status: 'error',
          progress: prev?.progress ?? 0,
          error: msg.error,
        });
        this.rejectPending(msg.taskId, new Error(msg.error));
        break;
      }
    }
  }

  private setState(info: TaskStatusInfo): void {
    this.states.set(info.taskId, info);
    this.emit('statusChange', { ...info });
    // отмена тоже завершает ожидающих getResult
    if (info.status === 'cancelled') {
      this.rejectPending(info.taskId, new Error('Задача отменена'));
    }
  }

  private resolvePending(taskId: TaskId, blob: Blob): void {
    this.pendingResults.get(taskId)?.forEach((p) => p.resolve(blob));
    this.pendingResults.delete(taskId);
  }

  private rejectPending(taskId: TaskId, err: Error): void {
    this.pendingResults.get(taskId)?.forEach((p) => p.reject(err));
    this.pendingResults.delete(taskId);
  }

  private generateId(): TaskId {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
      return crypto.randomUUID();
    }
    return `task-${Date.now().toString(36)}-${Math.floor(Math.random() * 1e9).toString(36)}`;
  }

  private async toWorkerSource(image: ImageInput): Promise<WorkerSource> {
    if (image instanceof ImageData) {
      // копируем буфер, чтобы не «увести» данные у вызывающего кода:
      // передаём копию как transferable
      const copy = image.data.slice().buffer;
      return {
        kind: 'imagedata',
        buffer: copy,
        width: image.width,
        height: image.height,
      };
    }
    // blob | file. heic декодируем здесь, потому что heic2any требует window
    // и не работает в воркере; в воркер уходит уже png
    let blob: Blob = image;
    let name = 'name' in image ? image.name : undefined;
    if (await isHeic(image)) {
      blob = await convertHeicToPng(image);
      name = name ? name.replace(/\.(heic|heif)$/i, '.png') : undefined;
    }
    const buffer = await blob.arrayBuffer();
    return {
      kind: 'encoded',
      buffer,
      mime: blob.type || 'image/png',
      name,
    };
  }
}
