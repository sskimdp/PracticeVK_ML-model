// декодирование heic/heif в главном потоке: heic2any использует window/dom
// и не работает в web worker, поэтому конвертируем heic → png здесь

/** эвристика heic по типу и имени файла */
function isHeicByMeta(file: Blob & { name?: string }): boolean {
  const type = (file.type || '').toLowerCase();
  if (type.includes('heic') || type.includes('heif')) return true;
  const name = (file.name ?? '').toLowerCase();
  return name.endsWith('.heic') || name.endsWith('.heif');
}

/** проверка iso-bmff сигнатуры: на смещении 4 — «ftyp», далее heic-бренд */
function hasHeicSignature(head: Uint8Array): boolean {
  if (head.length < 12) return false;
  const ascii = (i: number) =>
    String.fromCharCode(head[i], head[i + 1], head[i + 2], head[i + 3]);
  if (ascii(4) !== 'ftyp') return false;
  const brand = ascii(8);
  return ['heic', 'heix', 'heim', 'heis', 'hevc', 'mif1', 'msf1', 'heif'].includes(brand);
}

/** определяет, является ли файл heic, по мете и сигнатуре */
export async function isHeic(file: Blob & { name?: string }): Promise<boolean> {
  if (isHeicByMeta(file)) return true;
  try {
    const head = new Uint8Array(await file.slice(0, 12).arrayBuffer());
    return hasHeicSignature(head);
  } catch {
    return false;
  }
}

/** конвертирует heic-blob в png-blob, heic2any грузится лениво */
export async function convertHeicToPng(blob: Blob): Promise<Blob> {
  try {
    const { default: heic2any } = await import('heic2any');
    const out = await heic2any({ blob, toType: 'image/png' });
    return Array.isArray(out) ? out[0] : out;
  } catch (err) {
    throw new Error(
      `Не удалось декодировать HEIC: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}
