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
      outputCategoryMask: true,
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
    const categoryMask = result.categoryMask
    if ((!masks || masks.length === 0) && !categoryMask) {
      result.close?.()
      return canvas
    }

    const confidenceData = masks?.map((mask) => mask.getAsFloat32Array())
    const categoryData = categoryMask?.getAsUint8Array()
    const sourceWidth = masks?.[0]?.width ?? categoryMask?.width ?? MASK_WIDTH
    const sourceHeight = masks?.[0]?.height ?? categoryMask?.height ?? MASK_HEIGHT
    const out = imageData.data
    for (let y = 0, j = 0; y < MASK_HEIGHT; y += 1) {
      const sy = Math.min(sourceHeight - 1, Math.floor((y * sourceHeight) / MASK_HEIGHT))
      for (let x = 0; x < MASK_WIDTH; x += 1, j += 4) {
        const sx = Math.min(sourceWidth - 1, Math.floor((x * sourceWidth) / MASK_WIDTH))
        const sourceIndex = sy * sourceWidth + sx
        let foreground = 0
        if (confidenceData && confidenceData.length > 1) {
          const personLayerEnd = Math.min(confidenceData.length, 5)
          for (let layer = 1; layer < personLayerEnd; layer += 1) {
            foreground = Math.max(foreground, confidenceData[layer][sourceIndex])
          }
        } else if (confidenceData?.[0]) {
          foreground = 1 - confidenceData[0][sourceIndex]
        }

        const category = categoryData?.[sourceIndex] ?? 0
        if (category > 0 && category < 5) {
          foreground = Math.max(foreground, 0.96)
        }
        const v = Math.max(0, Math.min(255, Math.round(foreground * 255)))
        out[j] = v
        out[j + 1] = v
        out[j + 2] = v
        out[j + 3] = 255
      }
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
