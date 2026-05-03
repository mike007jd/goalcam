import { Camera, LoaderCircle, Pause, SlidersHorizontal, Sparkles } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { requestMacBookCamera, stopMediaStream } from './lib/camera'
import { drawDemoFrame } from './lib/demoSource'
import { MASK_HEIGHT, MASK_WIDTH, Segmenter } from './lib/segmenter'
import {
  DEFAULT_SETTINGS,
  RENDER_HEIGHT,
  RENDER_WIDTH,
  type EngineStats,
  type FxSettings,
  type RunMode,
} from './lib/types'
import { WebGpuFxRenderer } from './lib/webgpuRenderer'

type RendererStatus = {
  state: 'booting' | 'ready' | 'error'
  message: string
}

const INITIAL_STATS: EngineStats = {
  fps: 0,
  frames: 0,
  renderer: 'WebGPU',
}

const BODY_FX_OPTIONS: Array<{ label: string; value: FxSettings['bodyFxMode'] }> = [
  { label: 'Invisible', value: 'none' },
  { label: 'Jelly', value: 'jelly' },
  { label: 'Water', value: 'water' },
  { label: 'Cloth', value: 'cloth' },
]

function App() {
  const outputCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const demoMaskCanvasRef = useRef<OffscreenCanvas | HTMLCanvasElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const rendererRef = useRef<WebGpuFxRenderer | null>(null)
  const segmenterRef = useRef<Segmenter | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const loopRef = useRef<number | null>(null)
  const renderFrameRef = useRef<(time: number) => void>(() => undefined)
  const updateStatsRef = useRef<(renderer: WebGpuFxRenderer, time: number) => void>(() => undefined)
  const settingsRef = useRef(DEFAULT_SETTINGS)
  const runModeRef = useRef<RunMode>('idle')
  const lastStatsAtRef = useRef(0)
  const lastFrameAtRef = useRef(0)
  const fpsRef = useRef(0)

  const [settings, setSettings] = useState<FxSettings>(DEFAULT_SETTINGS)
  const [runMode, setRunMode] = useState<RunMode>('idle')
  const [rendererStatus, setRendererStatus] = useState<RendererStatus>({
    state: 'booting',
    message: 'Starting WebGPU',
  })
  const [stats, setStats] = useState<EngineStats>(INITIAL_STATS)

  useEffect(() => {
    settingsRef.current = settings
  }, [settings])

  useEffect(() => {
    runModeRef.current = runMode
  }, [runMode])

  const getDemoMaskCanvas = useCallback((): OffscreenCanvas | HTMLCanvasElement => {
    if (!demoMaskCanvasRef.current) {
      demoMaskCanvasRef.current =
        typeof OffscreenCanvas === 'undefined'
          ? Object.assign(document.createElement('canvas'), { width: MASK_WIDTH, height: MASK_HEIGHT })
          : new OffscreenCanvas(MASK_WIDTH, MASK_HEIGHT)
    }
    return demoMaskCanvasRef.current
  }, [])

  const renderFrame = useCallback((time: number) => {
    const renderer = rendererRef.current
    const frameCanvas = frameCanvasRef.current
    const video = videoRef.current
    if (!renderer || !frameCanvas) {
      loopRef.current = null
      return
    }

    const mode = runModeRef.current
    let maskCanvas: OffscreenCanvas | HTMLCanvasElement | null

    if (mode === 'camera') {
      const ctx = frameCanvas.getContext('2d', { willReadFrequently: true })
      if (ctx && video?.readyState && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        ctx.drawImage(video, 0, 0, RENDER_WIDTH, RENDER_HEIGHT)
      }
      maskCanvas = segmenterRef.current?.segment(frameCanvas, time) ?? null
    } else {
      drawDemoFrame(frameCanvas, time)
      maskCanvas = drawDemoMask(getDemoMaskCanvas(), time)
    }

    renderer.render(frameCanvas, maskCanvas, settingsRef.current, time)
    updateStatsRef.current(renderer, time)
    loopRef.current = requestAnimationFrame(renderFrameRef.current)
  }, [getDemoMaskCanvas])

  useEffect(() => {
    updateStatsRef.current = updateStats
  })

  useEffect(() => {
    renderFrameRef.current = renderFrame
  }, [renderFrame])

  useEffect(() => {
    const outputCanvas = outputCanvasRef.current
    const frameCanvas = frameCanvasRef.current
    if (!outputCanvas || !frameCanvas) return

    frameCanvas.width = RENDER_WIDTH
    frameCanvas.height = RENDER_HEIGHT

    let cancelled = false
    const renderer = new WebGpuFxRenderer()
    const segmenter = new Segmenter()
    rendererRef.current = renderer
    segmenterRef.current = segmenter

    Promise.all([renderer.init(outputCanvas), segmenter.init()])
      .then(() => {
        if (cancelled) return
        setRendererStatus({ state: 'ready', message: renderer.label })
        setStats((current) => ({ ...current, renderer: renderer.label }))
        setRunMode('demo')
        runModeRef.current = 'demo'
        if (loopRef.current === null) {
          loopRef.current = requestAnimationFrame(renderFrame)
        }
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : 'Engine failed to start.'
        setRendererStatus({ state: 'error', message })
      })

    return () => {
      cancelled = true
      if (loopRef.current !== null) {
        cancelAnimationFrame(loopRef.current)
        loopRef.current = null
      }
      stopMediaStream(streamRef.current)
      streamRef.current = null
      segmenter.dispose()
      segmenterRef.current = null
      renderer.dispose()
    }
  }, [renderFrame])

  const startDemo = useCallback(() => {
    stopMediaStream(streamRef.current)
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setRunMode('demo')
    runModeRef.current = 'demo'
    if (loopRef.current === null) {
      loopRef.current = requestAnimationFrame(renderFrame)
    }
  }, [renderFrame])

  const startCamera = useCallback(async () => {
    try {
      const stream = await requestMacBookCamera()
      stopMediaStream(streamRef.current)
      streamRef.current = stream
      const video = videoRef.current
      if (video) {
        video.srcObject = stream
        await video.play()
      }
      setRunMode('camera')
      runModeRef.current = 'camera'
      if (loopRef.current === null) {
        loopRef.current = requestAnimationFrame(renderFrame)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Camera permission failed.'
      setRendererStatus((current) => ({ ...current, message }))
    }
  }, [renderFrame])

  const stopEngine = useCallback(() => {
    if (loopRef.current !== null) {
      cancelAnimationFrame(loopRef.current)
      loopRef.current = null
    }
    stopMediaStream(streamRef.current)
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setRunMode('idle')
    runModeRef.current = 'idle'
  }, [])

  const updateSetting = useCallback(<K extends keyof FxSettings>(key: K, value: FxSettings[K]) => {
    setSettings((current) => ({ ...current, [key]: value }))
  }, [])

  const statusTone = rendererStatus.state === 'ready' ? 'good' : rendererStatus.state
  const canUseCamera = rendererStatus.state === 'ready'
  const coreCopy = useMemo(() => 'ML person segmentation with local shader fill.', [])

  return (
    <main className="app-shell">
      <section className="stage-column" aria-label="InvisibleCam live compositor">
        <header className="topbar">
          <div>
            <h1>InvisibleCam</h1>
            <p>{coreCopy}</p>
          </div>
          <div className="topbar-actions">
            <button type="button" onClick={startDemo} className={runMode === 'demo' ? 'active' : ''}>
              <Sparkles size={16} />
              Demo
            </button>
            <button
              type="button"
              onClick={() => void startCamera()}
              disabled={!canUseCamera}
              className={runMode === 'camera' ? 'active' : ''}
            >
              <Camera size={16} />
              Camera
            </button>
            <button type="button" onClick={stopEngine}>
              <Pause size={16} />
              Stop
            </button>
          </div>
        </header>

        <div className="stage-frame">
          <canvas ref={outputCanvasRef} className="fx-canvas" aria-label="WebGPU FX output" />
          {rendererStatus.state !== 'ready' && (
            <div className="stage-blocker">
              <LoaderCircle size={28} className="spin" />
              <span>{rendererStatus.message}</span>
            </div>
          )}
        </div>

        <div className="meter-row" aria-label="engine status">
          <StatusPill label="Renderer" value={rendererStatus.message} tone={statusTone} />
          <StatusPill label="Input" value={runMode} tone={runMode === 'idle' ? 'idle' : 'good'} />
          <StatusPill label="Pipeline" value="Directional fill" tone="good" />
          <StatusPill label="FPS" value={stats.fps.toFixed(0)} tone="idle" />
          <StatusPill label="Mask" value="MediaPipe" tone="good" />
        </div>

        <video ref={videoRef} playsInline muted className="hidden-source" />
        <canvas ref={frameCanvasRef} className="hidden-source" />
      </section>

      <aside className="control-panel" aria-label="FX controls">
        <section className="panel-section">
          <div className="section-title">
            <SlidersHorizontal size={17} />
            <span>Compositor</span>
          </div>
          <SliderControl
            label="Invisibility strength"
            value={settings.opacity}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            onChange={(value) => updateSetting('opacity', value)}
          />
          <SliderControl
            label="Mask coverage"
            value={settings.followLock}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            onChange={(value) => updateSetting('followLock', value)}
          />
          <SliderControl
            label="Edge feather"
            value={settings.maskFeather}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            onChange={(value) => updateSetting('maskFeather', value)}
          />
          <SliderControl
            label="Mask stability"
            value={settings.maskStability}
            min={0}
            max={0.95}
            step={0.01}
            format={percent}
            onChange={(value) => updateSetting('maskStability', value)}
          />
        </section>

        <section className="panel-section">
          <div className="section-title">
            <Sparkles size={17} />
            <span>Stacked FX</span>
          </div>
          <div className="effect-list" role="radiogroup" aria-label="body effect">
            {BODY_FX_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                role="radio"
                aria-checked={settings.bodyFxMode === option.value}
                className={`effect-option ${settings.bodyFxMode === option.value ? 'active' : ''}`}
                onClick={() => updateSetting('bodyFxMode', option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </section>

        <section className="panel-section">
          <div className="section-title">
            <SlidersHorizontal size={17} />
            <span>Diagnostics</span>
          </div>
          <label className="field">
            <span>View</span>
            <select
              value={settings.debugView}
              onChange={(event) => updateSetting('debugView', event.target.value as FxSettings['debugView'])}
            >
              <option value="final">Final</option>
              <option value="matte">Matte</option>
              <option value="inpaint">Inpaint</option>
            </select>
          </label>
          <SliderControl
            label="Edge"
            value={settings.edgeGain}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            onChange={(value) => updateSetting('edgeGain', value)}
          />
        </section>
      </aside>
    </main>
  )

  function updateStats(renderer: WebGpuFxRenderer, time: number): void {
    const dt = time - lastFrameAtRef.current
    if (dt > 0 && dt < 250) {
      const instant = 1000 / dt
      fpsRef.current = fpsRef.current ? fpsRef.current * 0.9 + instant * 0.1 : instant
    }
    lastFrameAtRef.current = time

    if (time - lastStatsAtRef.current < 260) return
    lastStatsAtRef.current = time
    setStats({
      fps: fpsRef.current,
      frames: Math.round(time / 16.67),
      renderer: renderer.label,
    })
  }
}

function drawDemoMask(canvas: OffscreenCanvas | HTMLCanvasElement, time: number): OffscreenCanvas | HTMLCanvasElement {
  const ctx = canvas.getContext('2d')
  if (!ctx) return canvas

  const t = time * 0.001
  const cx = MASK_WIDTH * 0.5 + Math.sin(t * 1.4) * ((22 / RENDER_WIDTH) * MASK_WIDTH)
  const ground = MASK_HEIGHT * 0.86
  const x = (value: number) => (value / RENDER_WIDTH) * MASK_WIDTH
  const y = (value: number) => (value / RENDER_HEIGHT) * MASK_HEIGHT

  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, MASK_WIDTH, MASK_HEIGHT)
  ctx.filter = 'blur(2px)'
  ctx.fillStyle = '#fff'
  ctx.beginPath()
  ctx.ellipse(cx, y(RENDER_HEIGHT * 0.24), x(53), y(61), 0, 0, Math.PI * 2)
  ctx.fill()
  ctx.beginPath()
  ctx.roundRect(cx - x(77), y(RENDER_HEIGHT * 0.34), x(154), y(254), x(66))
  ctx.fill()
  ctx.lineWidth = y(42)
  ctx.lineCap = 'round'
  ctx.strokeStyle = '#fff'
  ctx.beginPath()
  ctx.moveTo(cx - x(88), y(RENDER_HEIGHT * 0.42))
  ctx.quadraticCurveTo(cx - x(156), y(RENDER_HEIGHT * 0.54), cx - x(130), y(RENDER_HEIGHT * 0.74))
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(cx + x(88), y(RENDER_HEIGHT * 0.42))
  ctx.quadraticCurveTo(cx + x(148), y(RENDER_HEIGHT * 0.56), cx + x(114), y(RENDER_HEIGHT * 0.74))
  ctx.stroke()
  ctx.lineWidth = y(48)
  ctx.beginPath()
  ctx.moveTo(cx - x(40), y(RENDER_HEIGHT * 0.72))
  ctx.lineTo(cx - x(64), ground)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(cx + x(40), y(RENDER_HEIGHT * 0.72))
  ctx.lineTo(cx + x(68), ground)
  ctx.stroke()
  ctx.filter = 'none'

  return canvas
}

function StatusPill({
  label,
  value,
  tone,
}: {
  label: string
  value: string | number
  tone: 'good' | 'loading' | 'error' | 'booting' | 'idle'
}) {
  return (
    <div className={`status-pill ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function SliderControl({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  format: (value: number) => string
  onChange: (value: number) => void
}) {
  return (
    <label className="slider-control">
      <span>
        {label}
        <strong>{format(value)}</strong>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  )
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`
}

export default App
