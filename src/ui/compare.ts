// слайдер сравнения до/после: слева оригинал, справа результат.
// нижний слой — canvas результата (обновляется в реальном времени),
// поверх обрезается оригинал, видимый слева.

export interface CompareController {
  destroy(): void;
}

export function mountCompare(
  host: HTMLElement,
  beforeUrl: string,
  afterEl: HTMLCanvasElement,
): CompareController {
  const wrap = document.createElement('div');
  wrap.className = 'compare';

  afterEl.classList.add('cmp-img', 'base');
  afterEl.setAttribute('draggable', 'false');

  const reveal = document.createElement('div');
  reveal.className = 'cmp-reveal';
  const beforeImg = document.createElement('img');
  beforeImg.className = 'cmp-img';
  beforeImg.src = beforeUrl;
  beforeImg.alt = 'До';
  beforeImg.draggable = false;
  reveal.appendChild(beforeImg);

  const tagL = tag('cmp-tag left', 'До');
  const tagR = tag('cmp-tag right', 'После');
  const handle = document.createElement('div');
  handle.className = 'cmp-handle';

  wrap.append(afterEl, reveal, tagL, tagR, handle);
  host.appendChild(wrap);

  const syncWidth = () => {
    beforeImg.style.width = `${wrap.clientWidth}px`;
  };
  const ro = new ResizeObserver(syncWidth);
  ro.observe(wrap);
  syncWidth();

  const setPos = (pct: number) => {
    const p = Math.max(0, Math.min(100, pct));
    reveal.style.width = `${p}%`;
    handle.style.left = `${p}%`;
  };
  setPos(50);

  const setFromClientX = (clientX: number) => {
    const rect = wrap.getBoundingClientRect();
    setPos(((clientX - rect.left) / rect.width) * 100);
  };

  let dragging = false;
  const onDown = (e: PointerEvent) => {
    e.preventDefault();
    dragging = true;
    wrap.setPointerCapture(e.pointerId);
    setFromClientX(e.clientX);
  };
  const onMove = (e: PointerEvent) => {
    if (!dragging) return;
    e.preventDefault();
    setFromClientX(e.clientX);
  };
  const onUp = () => {
    dragging = false;
  };

  wrap.addEventListener('pointerdown', onDown);
  wrap.addEventListener('pointermove', onMove);
  wrap.addEventListener('pointerup', onUp);
  wrap.addEventListener('pointercancel', onUp);

  return {
    destroy() {
      ro.disconnect();
      wrap.removeEventListener('pointerdown', onDown);
      wrap.removeEventListener('pointermove', onMove);
      wrap.removeEventListener('pointerup', onUp);
      wrap.removeEventListener('pointercancel', onUp);
      wrap.remove();
    },
  };
}

function tag(className: string, text: string): HTMLSpanElement {
  const el = document.createElement('span');
  el.className = className;
  el.textContent = text;
  return el;
}
