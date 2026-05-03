export const RENDER_WIDTH = 960
export const RENDER_HEIGHT = 540

export type RunMode = 'idle' | 'camera' | 'demo'
export type DebugView = 'final' | 'matte' | 'inpaint'

export type FxSettings = {
  opacity: number
  followLock: number
  maskFeather: number
  maskStability: number
  jelly: number
  water: number
  refraction: number
  edgeGain: number
  debugView: DebugView
}

export const DEFAULT_SETTINGS: FxSettings = {
  opacity: 0.96,
  followLock: 0.82,
  maskFeather: 0.4,
  maskStability: 0.6,
  jelly: 0,
  water: 0,
  refraction: 0,
  edgeGain: 0,
  debugView: 'final',
}

export type EngineStats = {
  fps: number
  frames: number
  renderer: string
}
