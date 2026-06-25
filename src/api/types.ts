// контракты api: общий договор между главным потоком (ui + фасад)
// и фоновым потоком (web worker)

/** идентификатор задачи обработки */
export type TaskId = string;

/** стадии жизненного цикла задачи */
export type TaskStatus =
  | 'queued'
  | 'decoding' // декодирование, в т.ч. heic
  | 'analyzing' // инференс модели, подбор параметров
  | 'applying' // применение коррекции к полному изображению
  | 'encoding'
  | 'done'
  | 'error'
  | 'cancelled';

/** коэффициенты коррекции, которые подбирает модель */
export interface CorrectionParams {
  /** яркость: множитель, 1.0 — без изменений */
  brightness: number;
  /** контраст: множитель, 1.0 — без изменений */
  contrast: number;
  /** насыщенность: множитель, 1.0 — без изменений */
  saturation: number;
}

/** снимок состояния задачи */
export interface TaskStatusInfo {
  taskId: TaskId;
  status: TaskStatus;
  /** прогресс выполнения, 0..1 */
  progress: number;
  /** подобранные параметры, появляются после стадии analyzing */
  params?: CorrectionParams;
  /** текст ошибки, только при status === 'error' */
  error?: string;
}

/** терминальные статусы — после них задача не меняется */
export const TERMINAL_STATUSES: ReadonlySet<TaskStatus> = new Set<TaskStatus>([
  'done',
  'error',
  'cancelled',
]);

// протокол обмена сообщениями: главный поток и web worker

/** источник изображения, передаваемый в worker через transferable buffer */
export type WorkerSource =
  | {
      kind: 'encoded';
      /** сырые байты файла (jpg/png/heic/bmp) */
      buffer: ArrayBuffer;
      mime: string;
      name?: string;
    }
  | {
      kind: 'imagedata';
      buffer: ArrayBuffer; // rgba-данные
      width: number;
      height: number;
    };

/** сообщения: главный поток → worker */
export type RequestMessage =
  | { type: 'init'; modelUrl: string }
  | { type: 'create'; taskId: TaskId; source: WorkerSource }
  | { type: 'cancel'; taskId: TaskId }
  // повторное применение коррекции с ручными параметрами, полноразмерно
  | { type: 'reapply'; reqId: string; taskId: TaskId; params: CorrectionParams };

/** сообщения: worker → главный поток */
export type ResponseMessage =
  | { type: 'status'; info: TaskStatusInfo }
  // preview — декодированный оригинал, уменьшенный, для ui и ручного превью
  | { type: 'result'; taskId: TaskId; blob: Blob; preview: Blob; params: CorrectionParams }
  | { type: 'reapplyResult'; reqId: string; blob: Blob }
  | { type: 'error'; taskId: TaskId; error: string };

/** карта событий, на которые можно подписаться у фасада */
export interface ApiEventMap {
  /** возникает при изменении статуса или прогресса любой задачи */
  statusChange: TaskStatusInfo;
}
