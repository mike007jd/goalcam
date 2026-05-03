import { RENDER_HEIGHT, RENDER_WIDTH } from './types'

export function drawDemoFrame(canvas: HTMLCanvasElement, time: number): void {
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return

  const w = canvas.width
  const h = canvas.height
  const t = time * 0.001
  const gradient = ctx.createLinearGradient(0, 0, w, h)
  gradient.addColorStop(0, '#e7edf0')
  gradient.addColorStop(0.45, '#c8d4d8')
  gradient.addColorStop(1, '#9eb0b3')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, w, h)

  ctx.fillStyle = '#b7c3c5'
  for (let x = -w; x < w * 2; x += 84) {
    const skew = Math.sin(t * 0.45 + x * 0.01) * 18
    ctx.fillRect(x + skew, h * 0.68, 36, h * 0.34)
  }

  ctx.strokeStyle = 'rgba(49, 66, 70, 0.24)'
  ctx.lineWidth = 1
  for (let y = 36; y < h; y += 54) {
    ctx.beginPath()
    ctx.moveTo(0, y + Math.sin(t + y) * 5)
    ctx.lineTo(w, y + Math.cos(t * 0.7 + y) * 5)
    ctx.stroke()
  }

  ctx.fillStyle = '#1b2326'
  const sway = Math.sin(t * 1.4) * 22
  const cx = w * 0.5 + sway
  const ground = h * 0.86
  ctx.beginPath()
  ctx.ellipse(cx, h * 0.24, 50, 58, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.beginPath()
  ctx.roundRect(cx - 74, h * 0.34, 148, 250, 62)
  ctx.fill()
  ctx.lineWidth = 38
  ctx.lineCap = 'round'
  ctx.strokeStyle = '#1b2326'
  ctx.beginPath()
  ctx.moveTo(cx - 86, h * 0.42)
  ctx.quadraticCurveTo(cx - 154, h * 0.54, cx - 128, h * 0.74)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(cx + 86, h * 0.42)
  ctx.quadraticCurveTo(cx + 146, h * 0.56, cx + 112, h * 0.74)
  ctx.stroke()
  ctx.lineWidth = 44
  ctx.beginPath()
  ctx.moveTo(cx - 38, h * 0.72)
  ctx.lineTo(cx - 62, ground)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(cx + 38, h * 0.72)
  ctx.lineTo(cx + 66, ground)
  ctx.stroke()
}

export function makeDemoMask(time: number): Uint8ClampedArray {
  const maskCanvas = document.createElement('canvas')
  maskCanvas.width = RENDER_WIDTH
  maskCanvas.height = RENDER_HEIGHT
  const ctx = maskCanvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return new Uint8ClampedArray(RENDER_WIDTH * RENDER_HEIGHT)

  const t = time * 0.001
  const cx = RENDER_WIDTH * 0.5 + Math.sin(t * 1.4) * 22
  const ground = RENDER_HEIGHT * 0.86

  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, RENDER_WIDTH, RENDER_HEIGHT)
  ctx.filter = 'blur(5px)'
  ctx.fillStyle = '#fff'
  ctx.beginPath()
  ctx.ellipse(cx, RENDER_HEIGHT * 0.24, 53, 61, 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.beginPath()
  ctx.roundRect(cx - 77, RENDER_HEIGHT * 0.34, 154, 254, 66)
  ctx.fill()
  ctx.lineWidth = 42
  ctx.lineCap = 'round'
  ctx.strokeStyle = '#fff'
  ctx.beginPath()
  ctx.moveTo(cx - 88, RENDER_HEIGHT * 0.42)
  ctx.quadraticCurveTo(cx - 156, RENDER_HEIGHT * 0.54, cx - 130, RENDER_HEIGHT * 0.74)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(cx + 88, RENDER_HEIGHT * 0.42)
  ctx.quadraticCurveTo(cx + 148, RENDER_HEIGHT * 0.56, cx + 114, RENDER_HEIGHT * 0.74)
  ctx.stroke()
  ctx.lineWidth = 48
  ctx.beginPath()
  ctx.moveTo(cx - 40, RENDER_HEIGHT * 0.72)
  ctx.lineTo(cx - 64, ground)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(cx + 40, RENDER_HEIGHT * 0.72)
  ctx.lineTo(cx + 68, ground)
  ctx.stroke()
  ctx.filter = 'none'

  const imageData = ctx.getImageData(0, 0, RENDER_WIDTH, RENDER_HEIGHT).data
  const mask = new Uint8ClampedArray(RENDER_WIDTH * RENDER_HEIGHT)
  for (let i = 0, j = 0; i < imageData.length; i += 4, j += 1) {
    mask[j] = imageData[i]
  }
  return mask
}
