import { Camera, LoaderCircle, Pause, RefreshCw, SlidersHorizontal, Sparkles } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { requestMacBookCamera, stopMediaStream } from './lib/camera'
import { drawDemoFrame } from './lib/demoSource'
import { Segmenter } from './lib/segmenter'
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

function App() {
  const outputCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null)
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

  const renderFrame = useCallback((time: number) => {
    const renderer = rendererRef.current
    const frameCanvas = frameCanvasRef.current
    const video = videoRef.current
    if (!renderer || !frameCanvas) {
      loopRef.current = null
      return
    }

    if (runModeRef.current === 'camera') {
      const ctx = frameCanvas.getContext('2d', { willReadFrequently: true })
      if (ctx && video?.readyState && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        ctx.drawImage(video, 0, 0, RENDER_WIDTH, RENDER_HEIGHT)
      }
    } else {
      drawDemoFrame(frameCanvas, time)
    }

    const maskCanvas = segmenterRef.current?.segment(frameCanvas, time) ?? null
    renderer.render(frameCanvas, maskCanvas, settingsRef.current, time)
    updateStatsRef.current(renderer, time)
    loopRef.current = requestAnimationFrame(renderFrameRef.current)
  }, [])

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

    renderer
      .init(outputCanvas)
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
        const message = error instanceof Error ? error.message : 'WebGPU failed to start.'
        setRendererStatus({ state: 'error', message })
      })

    segmenter.init().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : 'Segmenter failed to load.'
      setRendererStatus((current) => ({ ...current, message: `Segmenter: ${message}` }))
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
    resetTrackingState()
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
      resetTrackingState()
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
    resetTrackingState()
    setRunMode('idle')
    runModeRef.current = 'idle'
  }, [])

  const resetPlate = useCallback(() => {
    rendererRef.current?.resetPlate()
  }, [])

  const updateSetting = useCallback(<K extends keyof FxSettings>(key: K, value: FxSettings[K]) => {
    setSettings((current) => ({ ...current, [key]: value }))
  }, [])

  const statusTone = rendererStatus.state === 'ready' ? 'good' : rendererStatus.state
  const canUseCamera = rendererStatus.state === 'ready'
  const coreCopy = useMemo(
    () => 'ML person segmentation with temporal background reconstruction and shader inpainting.',
    [],
  )

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
          <StatusPill label="Pipeline" value="ML + WebGPU" tone="good" />
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
            label="Mask dilation"
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
          <SliderControl
            label="Inpaint fallback"
            value={settings.inpaintFallback}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            onChange={(value) => updateSetting('inpaintFallback', value)}
          />
          <SliderControl
            label="Plate learning"
            value={settings.plateLearning}
            min={0.005}
            max={0.18}
            step={0.005}
            format={decimal}
            onChange={(value) => updateSetting('plateLearning', value)}
          />
          <button type="button" onClick={resetPlate} className="secondary-action">
            <RefreshCw size={16} />
            Reset plate
          </button>
        </section>

        <section className="panel-section">
          <div className="section-title">
            <Sparkles size={17} />
            <span>Stacked FX</span>
          </div>
          <SliderControl
            label="Jelly"
            value={settings.jelly}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            onChange={(value) => updateSetting('jelly', value)}
          />
          <SliderControl
            label="Water"
            value={settings.water}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            onChange={(value) => updateSetting('water', value)}
          />
          <SliderControl
            label="Cloth"
            value={settings.cloth}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            onChange={(value) => updateSetting('cloth', value)}
          />
          <SliderControl
            label="Refraction"
            value={settings.refraction}
            min={0}
            max={1.4}
            step={0.01}
            format={decimal}
            onChange={(value) => updateSetting('refraction', value)}
          />
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
              <option value="plate">Plate</option>
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

  function resetTrackingState(): void {
    rendererRef.current?.resetPlate()
  }
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

function decimal(value: number): string {
  return value.toFixed(2)
}

export default App
