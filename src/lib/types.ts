export const RENDER_WIDTH = 960
export const RENDER_HEIGHT = 540

export type RunMode = 'idle' | 'camera' | 'demo'
export type DebugView = 'final' | 'matte' | 'inpaint'
export type BodyFxMode = 'none' | 'jelly' | 'water' | 'cloth'
export type MaskPart = 'hair' | 'bodySkin' | 'faceSkin' | 'clothes' | 'accessories'

export const MASK_PART_OPTIONS: Array<{ label: string; value: MaskPart; category: number }> = [
  { label: 'Hair', value: 'hair', category: 1 },
  { label: 'Body skin', value: 'bodySkin', category: 2 },
  { label: 'Face skin', value: 'faceSkin', category: 3 },
  { label: 'Clothes', value: 'clothes', category: 4 },
  { label: 'Accessories', value: 'accessories', category: 5 },
]

export type FxSettings = {
  opacity: number
  followLock: number
  maskFeather: number
  maskStability: number
  maskParts: MaskPart[]
  bodyFxMode: BodyFxMode
  edgeGain: number
  debugView: DebugView
}

export const DEFAULT_SETTINGS: FxSettings = {
  opacity: 0.96,
  followLock: 0.66,
  maskFeather: 0.4,
  maskStability: 0.6,
  maskParts: ['hair', 'bodySkin', 'faceSkin', 'clothes'],
  bodyFxMode: 'none',
  edgeGain: 0,
  debugView: 'final',
}

export type EngineStats = {
  fps: number
  frames: number
  renderer: string
}
