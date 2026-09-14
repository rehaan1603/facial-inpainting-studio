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
  root.InpaintingFraming = {frameGeometry, maskGeometry};
  if (typeof module !== 'undefined') module.exports = root.InpaintingFraming;
})(globalThis);
