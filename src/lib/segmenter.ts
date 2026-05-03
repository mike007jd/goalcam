import {
  FilesetResolver,
  ImageSegmenter,
  type ImageSegmenterResult,
} from '@mediapipe/tasks-vision'

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'

export const MASK_WIDTH = 256
export const MASK_HEIGHT = 256

export class Segmenter {
  private segmenter: ImageSegmenter | null = null
  private maskCanvas: OffscreenCanvas | null = null
  private maskCtx: OffscreenCanvasRenderingContext2D | null = null
  private maskImageData: ImageData | null = null
  private lastTimestamp = 0

  async init(): Promise<void> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_BASE)
    this.segmenter = await ImageSegmenter.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    })
    this.maskCanvas = new OffscreenCanvas(MASK_WIDTH, MASK_HEIGHT)
    const ctx = this.maskCanvas.getContext('2d')
    if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable.')
    this.maskCtx = ctx
    this.maskImageData = ctx.createImageData(MASK_WIDTH, MASK_HEIGHT)
  }

  segment(source: HTMLCanvasElement | HTMLVideoElement, timestamp: number): OffscreenCanvas | null {
    const segmenter = this.segmenter
    const canvas = this.maskCanvas
    const ctx = this.maskCtx
    const imageData = this.maskImageData
    if (!segmenter || !canvas || !ctx || !imageData) return null

    const ts = timestamp <= this.lastTimestamp ? this.lastTimestamp + 1 : Math.round(timestamp)
    this.lastTimestamp = ts

    const result: ImageSegmenterResult = segmenter.segmentForVideo(source, ts)
    const masks = result.confidenceMasks
    if (!masks || masks.length === 0) {
      result.close?.()
      return canvas
    }

    const bg = masks[0]
    const data = bg.getAsFloat32Array()
    const out = imageData.data
    for (let i = 0, j = 0; i < data.length; i += 1, j += 4) {
      const v = Math.max(0, Math.min(255, Math.round((1 - data[i]) * 255)))
      out[j] = v
      out[j + 1] = v
      out[j + 2] = v
      out[j + 3] = 255
    }
    ctx.putImageData(imageData, 0, 0)
    result.close?.()
    return canvas
  }

  dispose(): void {
    this.segmenter?.close()
    this.segmenter = null
    this.maskCanvas = null
    this.maskCtx = null
    this.maskImageData = null
  }
}
