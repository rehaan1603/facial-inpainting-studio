(function (root) {
  function frameGeometry(width, height, mode, size = 512) {
    if (!(width > 0 && height > 0) || !['crop', 'fit'].includes(mode)) throw new Error('Invalid image framing.');
    if (mode === 'crop') {
      const side = Math.min(width, height);
      return [(width - side) / 2, (height - side) / 2, side, side, 0, 0, size, size];
    }
    const scale = size / Math.max(width, height);
    return [0, 0, width, height, (size - width * scale) / 2, (size - height * scale) / 2, width * scale, height * scale];
  }
  function maskGeometry(width, height, sourceWidth, sourceHeight, mode) {
    if (width === sourceWidth && height === sourceHeight) return frameGeometry(width, height, mode);
    if (width === 512 && height === 512) return [0, 0, 512, 512, 0, 0, 512, 512];
    throw new Error('Use a mask matching the original photo dimensions, or a 512 × 512 mask matching the displayed frame.');
  }
  function rasterizeMask(rgba, width, height, geometry, size = 512) {
    if (rgba.length !== width * height * 4) throw new Error('Invalid mask pixel data.');
    const [sx, sy, sw, sh, dx, dy, dw, dh] = geometry;
    const output = new Uint8Array(size * size);
    // Keep every marked source pixel that overlaps the frame, including thin damage.
    for (let y = Math.max(0, Math.floor(sy)); y < Math.min(height, Math.ceil(sy + sh)); y++) {
      const top = Math.max(y, sy), bottom = Math.min(y + 1, sy + sh);
      if (bottom <= top) continue;
      const y0 = Math.max(0, Math.floor(dy + (top - sy) * dh / sh));
      const y1 = Math.min(size, Math.ceil(dy + (bottom - sy) * dh / sh));
      for (let x = Math.max(0, Math.floor(sx)); x < Math.min(width, Math.ceil(sx + sw)); x++) {
        const i = 4 * (y * width + x);
        if ((rgba[i] + rgba[i + 1] + rgba[i + 2]) / 3 * rgba[i + 3] / 255 < 128) continue;
        const left = Math.max(x, sx), right = Math.min(x + 1, sx + sw);
        if (right <= left) continue;
        const x0 = Math.max(0, Math.floor(dx + (left - sx) * dw / sw));
        const x1 = Math.min(size, Math.ceil(dx + (right - sx) * dw / sw));
        for (let yy = y0; yy < y1; yy++) output.fill(255, yy * size + x0, yy * size + x1);
      }
    }
    return output;
  }
  root.InpaintingFraming = {frameGeometry, maskGeometry, rasterizeMask};
  if (typeof module !== 'undefined') module.exports = root.InpaintingFraming;
})(globalThis);
