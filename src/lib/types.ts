export const RENDER_WIDTH = 960
export const RENDER_HEIGHT = 540

export type RunMode = 'idle' | 'camera' | 'demo'
export type DebugView = 'final' | 'matte' | 'plate' | 'inpaint'

export type FxSettings = {
  opacity: number
  plateLearning: number
  followLock: number
  maskFeather: number
  maskStability: number
  inpaintFallback: number
  jelly: number
  water: number
  cloth: number
  refraction: number
  edgeGain: number
  debugView: DebugView
}

export const DEFAULT_SETTINGS: FxSettings = {
  opacity: 0.96,
  plateLearning: 0.055,
  followLock: 0.82,
  maskFeather: 0.4,
  maskStability: 0.6,
  inpaintFallback: 0.85,
  jelly: 0.38,
  water: 0.3,
  cloth: 0.22,
  refraction: 0.72,
  edgeGain: 0.2,
  debugView: 'final',
}

export type EngineStats = {
  fps: number
  frames: number
  renderer: string
}
