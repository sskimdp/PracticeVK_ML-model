// главный компонент интерфейса: авто-улучшение ml-моделью,
// ручная настройка слайдерами, сравнение до/после, скачивание, отмена, ошибки.

import { ImageEnhancerAPI, type ImageInput } from '../api/ImageEnhancerAPI.ts';
import type { CorrectionParams, TaskId, TaskStatusInfo } from '../api/types.ts';
import { applyCorrection } from '../worker/correction.ts';
import { STATUS_LABELS, formatBytes, formatMegapixels } from './labels.ts';
import { mountCompare, type CompareController } from './compare.ts';

const ACCEPT = '.jpg,.jpeg,.png,.bmp,.heic,.heif,image/*';
const FACTOR_MIN = 0.4;
const FACTOR_MAX = 2.5;

// инлайновые svg-иконки, чтобы не тянуть внешние зависимости
const ICON_UPLOAD =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></svg>';
const ICON_SPARKLE =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 4.5L18 9.3l-4.2 1.8L12 15.6 10.2 11.1 6 9.3l4.2-1.8z"/><path d="M19 14l.7 1.8L21.5 16l-1.8.7L19 18.5 18.3 16.7 16.5 16l1.8-.7z"/></svg>';
const ICON_DOWNLOAD =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v12M12 16l-4-4M12 16l4-4"/><path d="M4 20h16"/></svg>';

export class App {
  private api = new ImageEnhancerAPI();
  private root: HTMLElement;

  private file: File | null = null;
  private selectedUrl: string | null = null; // превью выбранного файла до обработки
  private beforeUrl: string | null = null; // декодированный оригинал для сравнения
  private taskId: TaskId | null = null;
  private busy = false;

  private compare: CompareController | null = null;
  private afterCanvas: HTMLCanvasElement | null = null;
  private previewData: ImageData | null = null; // уменьшенный оригинал для live-превью
  private autoParams: CorrectionParams = { brightness: 1, contrast: 1, saturation: 1 };
  private sliderParams: CorrectionParams = { brightness: 1, contrast: 1, saturation: 1 };
  private intensity = 1;
  private autoBlob: Blob | null = null;
  private rafScheduled = false;

  private els!: {
    dropzone: HTMLDivElement;
    fileInput: HTMLInputElement;
    enhanceBtn: HTMLButtonElement;
    cancelBtn: HTMLButtonElement;
    downloadBtn: HTMLButtonElement;
    resetBtn: HTMLButtonElement;
    meta: HTMLSpanElement;
    progressWrap: HTMLDivElement;
    progressLabel: HTMLSpanElement;
    progressPct: HTMLSpanElement;
    progressFill: HTMLDivElement;
    compareHost: HTMLDivElement;
    adjustCard: HTMLDivElement;
    sBright: HTMLInputElement;
    sContrast: HTMLInputElement;
    sSat: HTMLInputElement;
    sIntensity: HTMLInputElement;
    vBright: HTMLSpanElement;
    vContrast: HTMLSpanElement;
    vSat: HTMLSpanElement;
    vIntensity: HTMLSpanElement;
    autoResetBtn: HTMLButtonElement;
    errorMsg: HTMLDivElement;
  };

  constructor(root: HTMLElement) {
    this.root = root;
    this.render();
    this.bind();
    this.api.on('statusChange', (info) => this.onStatusChange(info));
  }

  private render(): void {
    this.root.innerHTML = `
      <header>
        <h1>Улучшение изображений</h1>
        <p>ML-модель подбирает яркость, контраст и насыщенность – обработка прямо в браузере.</p>
      </header>

      <section class="card">
        <div class="dropzone" id="dropzone">
          <div class="dz-icon">${ICON_UPLOAD}</div>
          <div class="dz-title">Перетащите изображение</div>
          <div>или нажмите, чтобы выбрать файл</div>
          <div class="formats">JPG · PNG · HEIC · BMP · до 15 Мпк</div>
          <input type="file" id="fileInput" accept="${ACCEPT}" hidden />
        </div>

        <div class="controls">
          <button class="btn btn-primary" id="enhanceBtn" disabled>${ICON_SPARKLE}Улучшить</button>
          <button class="btn btn-danger hidden" id="cancelBtn">Отменить</button>
          <button class="btn btn-primary hidden" id="downloadBtn">${ICON_DOWNLOAD}Скачать результат</button>
          <button class="btn btn-secondary hidden" id="resetBtn">Сбросить</button>
          <span class="meta" id="meta"></span>
        </div>

        <div class="progress-wrap faded hidden" id="progressWrap">
          <div class="progress-head">
            <span id="progressLabel">–</span>
            <span id="progressPct">0%</span>
          </div>
          <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
        </div>

        <div class="error-msg hidden" id="errorMsg"></div>
      </section>

      <section class="card compare-host faded hidden" id="compareHost"></section>

      <section class="card adjust faded hidden" id="adjustCard">
        <div class="adjust-head">
          <h2>Тонкая настройка</h2>
          <button class="btn btn-secondary btn-sm" id="autoResetBtn">Сбросить к авто</button>
        </div>
        ${this.sliderRow('sBright', 'Яркость', FACTOR_MIN, FACTOR_MAX, 0.01)}
        ${this.sliderRow('sContrast', 'Контраст', FACTOR_MIN, FACTOR_MAX, 0.01)}
        ${this.sliderRow('sSat', 'Насыщенность', FACTOR_MIN, FACTOR_MAX, 0.01)}
        ${this.sliderRow('sIntensity', 'Интенсивность', 0, 100, 1, '%')}
      </section>
    `;

    const q = <T extends HTMLElement>(s: string) => this.root.querySelector<T>(s)!;
    this.els = {
      dropzone: q('#dropzone'),
      fileInput: q('#fileInput'),
      enhanceBtn: q('#enhanceBtn'),
      cancelBtn: q('#cancelBtn'),
      downloadBtn: q('#downloadBtn'),
      resetBtn: q('#resetBtn'),
      meta: q('#meta'),
      progressWrap: q('#progressWrap'),
      progressLabel: q('#progressLabel'),
      progressPct: q('#progressPct'),
      progressFill: q('#progressFill'),
      compareHost: q('#compareHost'),
      adjustCard: q('#adjustCard'),
      sBright: q('#sBright'),
      sContrast: q('#sContrast'),
      sSat: q('#sSat'),
      sIntensity: q('#sIntensity'),
      vBright: q('#sBright-val'),
      vContrast: q('#sContrast-val'),
      vSat: q('#sSat-val'),
      vIntensity: q('#sIntensity-val'),
      autoResetBtn: q('#autoResetBtn'),
      errorMsg: q('#errorMsg'),
    };
  }

  private sliderRow(
    id: string,
    label: string,
    min: number,
    max: number,
    step: number,
    suffix = '',
  ): string {
    return `
      <div class="slider-row">
        <label for="${id}">${label}</label>
        <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" />
        <span class="slider-val" id="${id}-val">–${suffix}</span>
      </div>`;
  }

  private bind(): void {
    const e = this.els;
    e.dropzone.addEventListener('click', () => e.fileInput.click());
    e.fileInput.addEventListener('change', () => {
      const f = e.fileInput.files?.[0];
      if (f) void this.selectFile(f);
    });
    e.dropzone.addEventListener('dragover', (ev) => {
      ev.preventDefault();
      e.dropzone.classList.add('dragover');
    });
    e.dropzone.addEventListener('dragleave', () => e.dropzone.classList.remove('dragover'));
    e.dropzone.addEventListener('drop', (ev) => {
      ev.preventDefault();
      e.dropzone.classList.remove('dragover');
      const f = ev.dataTransfer?.files?.[0];
      if (f) void this.selectFile(f);
    });

    e.enhanceBtn.addEventListener('click', () => void this.enhance());
    e.cancelBtn.addEventListener('click', () => void this.cancel());
    e.downloadBtn.addEventListener('click', () => void this.download());
    e.resetBtn.addEventListener('click', () => this.reset());
    e.autoResetBtn.addEventListener('click', () => this.resetToAuto());

    e.sBright.addEventListener('input', () => this.onSlider());
    e.sContrast.addEventListener('input', () => this.onSlider());
    e.sSat.addEventListener('input', () => this.onSlider());
    e.sIntensity.addEventListener('input', () => this.onSlider());
  }

  private async selectFile(file: File): Promise<void> {
    this.clearError();
    this.file = file;
    this.revoke('selectedUrl');
    this.selectedUrl = URL.createObjectURL(file);
    const dims = await this.probeDimensions(this.selectedUrl).catch(() => null);

    const parts = [formatBytes(file.size)];
    if (dims) parts.push(formatMegapixels(dims.w, dims.h));
    this.els.meta.textContent = parts.join(' · ');

    // новый файл аннулирует прошлый результат — чистим состояние и ui
    this.setTheme('default');
    this.taskId = null;
    this.autoBlob = null;
    this.previewData = null;
    this.afterCanvas = null;
    this.revoke('beforeUrl');
    this.show(this.els.progressWrap, false);
    this.els.progressFill.style.width = '0%';
    this.els.progressFill.classList.remove('done', 'error');

    this.els.enhanceBtn.disabled = false;
    this.els.resetBtn.classList.remove('hidden');
    this.els.downloadBtn.classList.add('hidden');
    this.show(this.els.adjustCard, false);

    // превью может не отрисоваться для heic — это нормально,
    // после обработки покажем декодированный оригинал
    this.compare?.destroy();
    this.compare = null;
    if (dims) {
      this.els.compareHost.innerHTML = `<div class="compare"><img class="cmp-img" src="${this.selectedUrl}" alt="Исходное" draggable="false" /></div>`;
      this.show(this.els.compareHost, true);
    } else {
      this.show(this.els.compareHost, false);
    }
  }

  private probeDimensions(url: string): Promise<{ w: number; h: number }> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
      img.onerror = () => reject(new Error('no dims'));
      img.src = url;
    });
  }

  private async enhance(): Promise<void> {
    if (!this.file) return;
    this.clearError();
    this.setBusy(true);

    try {
      const input: ImageInput = this.file;
      const id = await this.api.createTask(input);
      this.taskId = id;

      this.autoBlob = await this.api.getResult(id);
      this.autoParams = this.api.getParams(id) ?? { brightness: 1, contrast: 1, saturation: 1 };

      // декодированный оригинал из воркера, работает и для heic
      const previewBlob = this.api.getPreview(id);
      if (previewBlob) {
        this.revoke('beforeUrl');
        this.beforeUrl = URL.createObjectURL(previewBlob);
        this.previewData = await this.blobToImageData(previewBlob);
      }

      this.sliderParams = { ...this.autoParams };
      this.intensity = 1;
      this.setupCompareAndAdjust();
      this.els.downloadBtn.classList.remove('hidden');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (msg !== 'Задача отменена') this.showError(msg);
    } finally {
      this.setBusy(false);
    }
  }

  private async cancel(): Promise<void> {
    if (this.taskId) await this.api.cancelTask(this.taskId);
  }

  private onStatusChange(info: TaskStatusInfo): void {
    if (info.taskId !== this.taskId || !this.busy) return;
    const e = this.els;
    this.show(e.progressWrap, true);
    e.progressLabel.textContent = STATUS_LABELS[info.status];
    e.progressPct.textContent = `${Math.round(info.progress * 100)}%`;
    e.progressFill.style.width = `${Math.round(info.progress * 100)}%`;
    e.progressFill.classList.toggle('done', info.status === 'done');
    e.progressFill.classList.toggle('error', info.status === 'error');

    if (info.status === 'done') this.setTheme('success');
    else if (info.status === 'error') this.setTheme('error');
    else if (info.status === 'cancelled') this.setTheme('default');
  }

  private setupCompareAndAdjust(): void {
    if (!this.previewData || !this.beforeUrl) return;
    this.compare?.destroy();

    this.afterCanvas = document.createElement('canvas');
    this.afterCanvas.width = this.previewData.width;
    this.afterCanvas.height = this.previewData.height;

    this.els.compareHost.innerHTML = '';
    this.compare = mountCompare(this.els.compareHost, this.beforeUrl, this.afterCanvas);
    this.show(this.els.compareHost, true);

    this.syncSliders();
    this.show(this.els.adjustCard, true);
    void this.renderAfter();
  }

  private effectiveParams(): CorrectionParams {
    const mix = (v: number) =>
      Math.min(FACTOR_MAX, Math.max(FACTOR_MIN, 1 + (v - 1) * this.intensity));
    return {
      brightness: mix(this.sliderParams.brightness),
      contrast: mix(this.sliderParams.contrast),
      saturation: mix(this.sliderParams.saturation),
    };
  }

  /** перерисовать слой «после» в live-превью на главном потоке по малой копии */
  private async renderAfter(): Promise<void> {
    if (!this.previewData || !this.afterCanvas) return;
    const eff = this.effectiveParams();
    const copy = new ImageData(
      new Uint8ClampedArray(this.previewData.data),
      this.previewData.width,
      this.previewData.height,
    );
    await applyCorrection(copy, eff, { onProgress: () => {}, shouldCancel: () => false });
    this.afterCanvas.getContext('2d')!.putImageData(copy, 0, 0);
  }

  private onSlider(): void {
    this.sliderParams = {
      brightness: parseFloat(this.els.sBright.value),
      contrast: parseFloat(this.els.sContrast.value),
      saturation: parseFloat(this.els.sSat.value),
    };
    this.intensity = parseInt(this.els.sIntensity.value, 10) / 100;
    this.updateSliderLabels();
    // коалесцируем перерисовку под кадр
    if (!this.rafScheduled) {
      this.rafScheduled = true;
      requestAnimationFrame(() => {
        this.rafScheduled = false;
        void this.renderAfter();
      });
    }
  }

  private syncSliders(): void {
    this.els.sBright.value = String(this.sliderParams.brightness);
    this.els.sContrast.value = String(this.sliderParams.contrast);
    this.els.sSat.value = String(this.sliderParams.saturation);
    this.els.sIntensity.value = String(Math.round(this.intensity * 100));
    this.updateSliderLabels();
  }

  private updateSliderLabels(): void {
    this.els.vBright.textContent = this.sliderParams.brightness.toFixed(2);
    this.els.vContrast.textContent = this.sliderParams.contrast.toFixed(2);
    this.els.vSat.textContent = this.sliderParams.saturation.toFixed(2);
    this.els.vIntensity.textContent = `${Math.round(this.intensity * 100)}%`;
  }

  private resetToAuto(): void {
    this.sliderParams = { ...this.autoParams };
    this.intensity = 1;
    this.syncSliders();
    void this.renderAfter();
  }

  private async download(): Promise<void> {
    if (!this.taskId) return;
    const eff = this.effectiveParams();
    let blob: Blob | null;
    // если параметры не меняли — берём готовый результат воркера
    if (this.isAutoSettings() && this.autoBlob) {
      blob = this.autoBlob;
    } else {
      const prev = this.els.downloadBtn.textContent;
      this.els.downloadBtn.disabled = true;
      this.els.downloadBtn.textContent = 'Готовлю файл…';
      try {
        blob = await this.api.reapplyTask(this.taskId, eff);
      } catch {
        this.showError('Не удалось подготовить файл');
        blob = null;
      } finally {
        this.els.downloadBtn.disabled = false;
        this.els.downloadBtn.textContent = prev;
      }
    }
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = this.makeOutputName(this.file?.name ?? 'image');
    a.click();
    URL.revokeObjectURL(url);
  }

  private isAutoSettings(): boolean {
    const eq = (a: number, b: number) => Math.abs(a - b) < 1e-3;
    return (
      this.intensity === 1 &&
      eq(this.sliderParams.brightness, this.autoParams.brightness) &&
      eq(this.sliderParams.contrast, this.autoParams.contrast) &&
      eq(this.sliderParams.saturation, this.autoParams.saturation)
    );
  }

  private makeOutputName(name: string): string {
    const dot = name.lastIndexOf('.');
    const base = dot > 0 ? name.slice(0, dot) : name;
    return `${base}_enhanced.png`;
  }

  private reset(): void {
    this.setTheme('default');
    this.clearError();

    // плавно убираем блоки результата
    this.show(this.els.progressWrap, false);
    this.show(this.els.adjustCard, false);
    this.show(this.els.compareHost, false);

    // состояние и кнопки — сразу
    this.file = null;
    this.taskId = null;
    this.previewData = null;
    this.afterCanvas = null;
    this.autoBlob = null;
    this.els.fileInput.value = '';
    this.els.meta.textContent = '';
    this.els.enhanceBtn.disabled = true;
    this.els.cancelBtn.classList.add('hidden');
    this.els.downloadBtn.classList.add('hidden');
    this.els.resetBtn.classList.add('hidden');

    // тяжёлую очистку откладываем до конца fade, чтобы контент не пропадал резко
    const compare = this.compare;
    const selUrl = this.selectedUrl;
    const befUrl = this.beforeUrl;
    this.compare = null;
    this.selectedUrl = null;
    this.beforeUrl = null;
    window.setTimeout(() => {
      compare?.destroy();
      if (selUrl) URL.revokeObjectURL(selUrl);
      if (befUrl) URL.revokeObjectURL(befUrl);
      // если за время fade выбрали новый файл — не стираем его превью
      if (this.file === null) this.els.compareHost.innerHTML = '';
    }, 600);
  }

  private setBusy(busy: boolean): void {
    this.busy = busy;
    this.els.enhanceBtn.disabled = busy || !this.file;
    this.els.cancelBtn.classList.toggle('hidden', !busy);
    this.els.resetBtn.classList.toggle('hidden', busy);
    this.els.dropzone.style.pointerEvents = busy ? 'none' : '';
    this.els.dropzone.style.opacity = busy ? '0.5' : '';
  }

  /** палитра фона зависит от состояния: синяя по умолчанию, зелёная при успехе, красная при ошибке */
  private setTheme(theme: 'default' | 'success' | 'error'): void {
    document.body.dataset.theme = theme;
  }

  /** плавное появление и скрытие блока через fade вместо резкого display:none */
  private show(el: HTMLElement, on: boolean): void {
    if (on) {
      el.classList.remove('hidden');
      // принудительный reflow фиксирует opacity:0 (display уже не none),
      // затем класс shown запускает переход; не зависит от raf,
      // который не срабатывает в фоновой вкладке
      void el.offsetWidth;
      el.classList.add('shown');
    } else {
      if (el.classList.contains('hidden')) return;
      el.classList.remove('shown');
      window.setTimeout(() => el.classList.add('hidden'), 550);
    }
  }

  private showError(msg: string): void {
    this.els.errorMsg.textContent = msg;
    this.els.errorMsg.classList.remove('hidden');
    this.setTheme('error');
  }
  private clearError(): void {
    this.els.errorMsg.textContent = '';
    this.els.errorMsg.classList.add('hidden');
  }

  private revoke(which: 'selectedUrl' | 'beforeUrl'): void {
    const url = this[which];
    if (url) URL.revokeObjectURL(url);
    this[which] = null;
  }

  private async blobToImageData(blob: Blob): Promise<ImageData> {
    const bmp = await createImageBitmap(blob);
    const canvas = document.createElement('canvas');
    canvas.width = bmp.width;
    canvas.height = bmp.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }
}
