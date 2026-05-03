# ML-Driven Invisibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current motion-difference invisibility algorithm with a Snapchat-class pipeline: MediaPipe person segmentation feeds a refined mask into a temporally learned background plate plus a push-pull pyramid inpainting fallback, composited with the existing FX layer.

**Architecture:** Five WebGPU passes per frame (segmenter → mask refine → plate learn → push-pull pyramid → composite). Shader code lives in `.wgsl` files imported via Vite `?raw`. The renderer becomes a multi-pass orchestrator. Paint mode is removed.

**Tech Stack:** TypeScript 6 / React 19 / Vite 8 / WebGPU / WGSL / `@mediapipe/tasks-vision`.

**Spec:** `docs/superpowers/specs/2026-05-03-ml-segmentation-invisibility-design.md`

**Verification model:** This project has no test framework. Each task verifies via `npm run lint`, `npm run build`, and a targeted manual check in `npm run dev`. The final task does an end-to-end smoke test against the spec's §8 verification list.

---

## Prerequisites

Before starting:

- [ ] **Confirm working directory:** `/Users/haoshengli/Seafile/WebWorkSpace/invisiblecam`
- [ ] **If not yet a git repository**, run `git init && git add -A && git commit -m "chore: initial baseline"` once before Task 1. Otherwise the per-task `git commit` steps will fail. If the user does not want git tracking, skip every `git add` / `git commit` step in this plan.
- [ ] **Confirm Node.js + npm install completes:** `npm install` (no errors).
- [ ] **Confirm baseline builds:** `npm run lint && npm run build` — both must pass before changes start.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/lib/segmenter.ts` | NEW | MediaPipe Tasks Vision wrapper. Loads model, runs `segmentForVideo`, returns mask as `OffscreenCanvas`. |
| `src/lib/shaders/maskRefine.wgsl` | NEW | Pass A: upsample raw mask, dilate, joint-bilateral filter, temporal EMA. |
| `src/lib/shaders/plateLearn.wgsl` | NEW | Pass B: mask-driven plate learning with confidence accumulation in alpha. |
| `src/lib/shaders/pushDown.wgsl` | NEW | Pass C, down half: 2×2 confidence-weighted average. |
| `src/lib/shaders/pushUp.wgsl` | NEW | Pass C, up half: blend coarser level into hole pixels. |
| `src/lib/shaders/composite.wgsl` | NEW | Pass D: full-screen triangle, plate+inpaint blend, FX displacement, edge glow, debug branches. |
| `src/lib/webgpuRenderer.ts` | REWRITTEN | Multi-pass orchestrator. Owns all textures and pipelines; consumes mask canvas from caller. |
| `src/lib/types.ts` | CHANGED | `FxSettings` reshape: drop paint fields, add `maskFeather`/`maskStability`/`inpaintFallback`, change `DebugView`. |
| `src/App.tsx` | CHANGED | Drop paint canvas + handlers + paint UI section. Add three new sliders. Wire segmenter into render loop. |
| `src/App.css` | CHANGED | Remove `.paint-canvas` and segmented-control styles for paint mode. |
| `README.md` | CHANGED | Remove "no ML/segmentation models" sentence; document new pipeline. |
| `package.json` | CHANGED | Add `@mediapipe/tasks-vision` dependency. |
| `vite.config.ts` | CHANGED (maybe) | Add `assetsInclude: ['**/*.wgsl']` if `?raw` import needs it (test first). |

Total new texture footprint at runtime (960×540 + pyramid): ~6 MB VRAM.

---

## Task 1 — Extract inline shaders to `.wgsl` files via Vite `?raw`

**Why first:** Validates the file convention before adding four new shaders. Pure refactor — zero behavior change.

**Files:**
- Create: `src/lib/shaders/legacyCompute.wgsl`
- Create: `src/lib/shaders/legacyRender.wgsl`
- Modify: `src/lib/webgpuRenderer.ts:324-560` (replace inline shader strings with imports)
- Modify (maybe): `vite.config.ts`

**Steps:**

- [ ] **Step 1.1: Create the shaders directory and copy current compute shader verbatim**

Create `src/lib/shaders/legacyCompute.wgsl` with the contents of the `computeShader` template literal in `src/lib/webgpuRenderer.ts` (lines 324–394 in the current file). Do NOT include the JavaScript `const computeShader = /* wgsl */` ` ` ` wrapper — only the WGSL source itself.

- [ ] **Step 1.2: Create `src/lib/shaders/legacyRender.wgsl`**

Copy the contents of the `renderShader` template literal in `src/lib/webgpuRenderer.ts` (lines 396–560 in the current file) into the new file. WGSL source only.

- [ ] **Step 1.3: Replace inline shader strings with `?raw` imports**

Edit `src/lib/webgpuRenderer.ts`:

At the top of the file (after the existing import on line 1), add:

```typescript
import computeShader from './shaders/legacyCompute.wgsl?raw'
import renderShader from './shaders/legacyRender.wgsl?raw'
```

Then delete the entire `const computeShader = /* wgsl */ ...` and `const renderShader = /* wgsl */ ...` blocks at the bottom of the file (everything from line 324 to the end of the file).

- [ ] **Step 1.4: Add WGSL type declaration for TypeScript**

Create `src/lib/shaders/wgsl.d.ts`:

```typescript
declare module '*.wgsl?raw' {
  const source: string
  export default source
}
```

- [ ] **Step 1.5: Verify**

Run:

```bash
npm run lint && npm run build && npm run dev
```

Open the printed localhost URL in a WebGPU-capable Chromium browser. Click **Demo** — the demo loop must render exactly as before (you should see the moving silhouette getting "invisibled" against the demo background). If it doesn't, the `?raw` import isn't working — try adding `assetsInclude: ['**/*.wgsl']` to `vite.config.ts`'s `defineConfig({ ... })` call.

- [ ] **Step 1.6: Commit**

```bash
git add src/lib/shaders/legacyCompute.wgsl src/lib/shaders/legacyRender.wgsl src/lib/shaders/wgsl.d.ts src/lib/webgpuRenderer.ts vite.config.ts
git commit -m "refactor: extract inline WGSL shaders to .wgsl files via Vite ?raw"
```

---

## Task 2 — Install MediaPipe Tasks Vision

**Files:**
- Modify: `package.json`, `package-lock.json`

**Steps:**

- [ ] **Step 2.1: Install the dependency**

```bash
npm install @mediapipe/tasks-vision@0.10.18
```

If the registry serves a newer 0.10.x patch, take it. We pin minor at 0.10 because the Tasks Vision API is stable there.

- [ ] **Step 2.2: Verify install**

```bash
ls node_modules/@mediapipe/tasks-vision/vision_bundle.mjs
```

Expected: file exists.

- [ ] **Step 2.3: Verify build still passes**

```bash
npm run lint && npm run build
```

Both must succeed (no usage yet, just dependency added).

- [ ] **Step 2.4: Commit**

```bash
git add package.json package-lock.json
git commit -m "chore: add @mediapipe/tasks-vision dependency"
```

---

## Task 3 — Write the segmenter wrapper (standalone, not yet wired)

**Files:**
- Create: `src/lib/segmenter.ts`

**Steps:**

- [ ] **Step 3.1: Write the segmenter class**

Create `src/lib/segmenter.ts`:

```typescript
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

    // selfie_multiclass_256x256: index 0 = background. Person confidence = 1 - background.
    const bg = masks[0]
    const data = bg.getAsFloat32Array()
    const out = imageData.data
    for (let i = 0, j = 0; i < data.length; i++, j += 4) {
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
```

- [ ] **Step 3.2: Verify build**

```bash
npm run lint && npm run build
```

Both must pass. Lint may complain about unused import — that's fine if the file is unreferenced; the import is exercised at compile time.

- [ ] **Step 3.3: Commit**

```bash
git add src/lib/segmenter.ts
git commit -m "feat: add MediaPipe person segmentation wrapper"
```

---

## Task 4 — Wire segmenter + add `rawMaskTexture` to renderer (debug-only path)

**Why staged like this:** We can validate the segmentation output visually before refining/learning anything. After this task, the **Matte** debug view shows the raw upsampled MediaPipe mask. The composite math is unchanged.

**Files:**
- Modify: `src/lib/webgpuRenderer.ts` (add `rawMaskTexture`, accept mask canvas in `render()`)
- Modify: `src/App.tsx` (instantiate `Segmenter`, call it per frame, pass mask canvas to renderer)
- Modify: `src/lib/shaders/legacyRender.wgsl` (temporary: route the `matte` debug view to sample `rawMaskTex` instead of computing motion)

**Steps:**

- [ ] **Step 4.1: Add `rawMaskTexture` and a setter in the renderer**

Edit `src/lib/webgpuRenderer.ts`. In the `WebGpuFxRenderer` class, add a private field next to the other textures:

```typescript
  private rawMaskTexture: GPUTexture | null = null
```

In `createTextures()`, after the existing texture creations, add:

```typescript
    this.rawMaskTexture = this.device.createTexture({
      size: { width: 256, height: 256 },
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
      label: 'raw-mediapipe-mask',
    })
```

In `dispose()`, add:

```typescript
    this.rawMaskTexture?.destroy()
```

- [ ] **Step 4.2: Change `render()` signature to accept the mask canvas**

In `src/lib/webgpuRenderer.ts`, change the signature of `render()`:

```typescript
  render(
    source: HTMLCanvasElement,
    paintCanvas: HTMLCanvasElement,
    maskSource: OffscreenCanvas | HTMLCanvasElement | null,
    settings: FxSettings,
    time: number,
  ): void {
```

(We keep `paintCanvas` for now — it's removed in Task 11.)

Inside `render()`, just after the existing `device.queue.copyExternalImageToTexture({ source }, { texture: videoTexture }, ...)` call, add:

```typescript
    if (maskSource && this.rawMaskTexture) {
      device.queue.copyExternalImageToTexture(
        { source: maskSource },
        { texture: this.rawMaskTexture },
        { width: 256, height: 256 },
      )
    }
```

- [ ] **Step 4.3: Bind `rawMaskTexture` into the render bind group**

In `createBindGroups()`, the render-side bind groups currently have entries for sampler, video, paint, plate, uniform (bindings 0..4). Add a sixth entry pointing to `rawMaskTexture` at binding 5. Replace the `this.renderBindGroups = plateViews.map(...)` block with:

```typescript
    const rawMaskView = this.rawMaskTexture!.createView()
    this.renderBindGroups = plateViews.map((plateView) =>
      this.device!.createBindGroup({
        layout: this.renderPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.sampler! },
          { binding: 1, resource: videoView },
          { binding: 2, resource: paintView },
          { binding: 3, resource: plateView },
          { binding: 4, resource: uniformResource },
          { binding: 5, resource: rawMaskView },
        ],
      }),
    )
```

Move the `if (...this.rawMaskTexture)` guard at the top of `createBindGroups()` up so the function returns early if `rawMaskTexture` is null.

- [ ] **Step 4.4: Add `rawMaskTex` binding in `legacyRender.wgsl`**

In `src/lib/shaders/legacyRender.wgsl`, after the line `@group(0) @binding(4) var<uniform> params: Params;`, add:

```wgsl
@group(0) @binding(5) var rawMaskTex: texture_2d<f32>;
```

In the `fragmentMain` function, replace the existing `matte` debug branch:

```wgsl
  if (params.debugView > 0.5 && params.debugView < 1.5) {
    return vec4f(vec3f(foreground), 1.0);
  }
```

with:

```wgsl
  if (params.debugView > 0.5 && params.debugView < 1.5) {
    let raw = textureSample(rawMaskTex, linearSampler, uv).r;
    return vec4f(vec3f(raw), 1.0);
  }
```

- [ ] **Step 4.5: Wire the segmenter in `App.tsx`**

Edit `src/App.tsx`:

Add the import at the top:

```typescript
import { Segmenter } from './lib/segmenter'
```

In the `App` function body, add a ref next to `rendererRef`:

```typescript
  const segmenterRef = useRef<Segmenter | null>(null)
```

In the `useEffect` block that creates the renderer (currently around line 102–144), after `rendererRef.current = renderer`, add segmenter creation in parallel with renderer init:

```typescript
    const segmenter = new Segmenter()
    segmenterRef.current = segmenter
    segmenter.init().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : 'Segmenter failed to load.'
      setRendererStatus((current) => ({ ...current, message: `Segmenter: ${message}` }))
    })
```

In the cleanup function of the same `useEffect`, add:

```typescript
      segmenter.dispose()
      segmenterRef.current = null
```

In `renderFrame` (currently around line 70–92), after the `if (runModeRef.current === 'camera') { ... } else { drawDemoFrame(...) }` block, before `renderer.render(...)`, add:

```typescript
    const maskCanvas = segmenterRef.current?.segment(frameCanvas, time) ?? null
```

Then change the `renderer.render(frameCanvas, paintCanvas, settingsRef.current, time)` line to:

```typescript
    renderer.render(frameCanvas, paintCanvas, maskCanvas, settingsRef.current, time)
```

- [ ] **Step 4.6: Verify**

```bash
npm run lint && npm run build && npm run dev
```

Open the localhost URL. Click **Demo**. Effect should still work as before (motion-based mask). Then in the **Diagnostics** panel, change **View** to **Matte** — you should now see the MediaPipe person mask (white-ish person blob on black) instead of the old motion mask. There may be a 1–3 second delay on first load while the model fetches.

Click **Camera**, allow camera permission. **Matte** view should now show YOUR silhouette as a clean white blob on black background.

If the mask is inverted (background white, person black), edit `src/lib/segmenter.ts` step 3.1 — invert the `(1 - data[i])` to `data[i]` (model class index 0 might be person in some model versions).

- [ ] **Step 4.7: Commit**

```bash
git add src/lib/webgpuRenderer.ts src/lib/shaders/legacyRender.wgsl src/App.tsx
git commit -m "feat: wire MediaPipe segmenter, expose raw mask via Matte debug view"
```

---

## Task 5 — Pass A: mask refine compute shader + ping-pong refined mask textures

**After this task:** The Matte debug view shows the *refined* mask (upsampled, dilated, edge-snapped, time-stable). The composite still uses the legacy motion math; we'll cut that over in Task 8.

**Files:**
- Create: `src/lib/shaders/maskRefine.wgsl`
- Modify: `src/lib/webgpuRenderer.ts` (add ping-pong refined mask textures, add Pass A pipeline + dispatch)
- Modify: `src/lib/types.ts` (add `maskFeather`, `maskStability` fields)
- Modify: `src/lib/shaders/legacyRender.wgsl` (route Matte view to refined mask, accept new binding)

**Steps:**

- [ ] **Step 5.1: Add `maskFeather` and `maskStability` to `FxSettings`**

Edit `src/lib/types.ts`. In the `FxSettings` type, add (next to `followLock`):

```typescript
  maskFeather: number
  maskStability: number
```

In `DEFAULT_SETTINGS`, add:

```typescript
  maskFeather: 0.4,
  maskStability: 0.6,
```

- [ ] **Step 5.2: Write the mask refine shader**

Create `src/lib/shaders/maskRefine.wgsl`:

```wgsl
struct Params {
  resolution: vec2f,
  maskFeather: f32,
  maskStability: f32,
  followLock: f32,
  pad0: f32, pad1: f32, pad2: f32,
};

@group(0) @binding(0) var rawMaskTex: texture_2d<f32>;
@group(0) @binding(1) var frameTex: texture_2d<f32>;
@group(0) @binding(2) var prevRefinedTex: texture_2d<f32>;
@group(0) @binding(3) var refinedOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var linearSampler: sampler;

fn luma(c: vec3f) -> f32 {
  return dot(c, vec3f(0.299, 0.587, 0.114));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= u32(params.resolution.x) || xy.y >= u32(params.resolution.y)) {
    return;
  }
  let pixel = vec2i(xy);
  let uv = (vec2f(xy) + 0.5) / params.resolution;

  // 1. Bilinear upsample raw mask to render-res
  var m = textureSampleLevel(rawMaskTex, linearSampler, uv, 0.0).r;

  // 2. Dilation (max filter) by followLock-controlled radius (up to ~14 px)
  let dilatePx = i32(round(params.followLock * 14.0));
  let res = vec2f(params.resolution);
  if (dilatePx > 0) {
    for (var dy = -dilatePx; dy <= dilatePx; dy = dy + 2) {
      for (var dx = -dilatePx; dx <= dilatePx; dx = dx + 2) {
        let s = uv + vec2f(f32(dx), f32(dy)) / res;
        m = max(m, textureSampleLevel(rawMaskTex, linearSampler, s, 0.0).r);
      }
    }
  }

  // 3. Joint bilateral / guided filter using frame luma
  let centerLuma = luma(textureLoad(frameTex, pixel, 0).rgb);
  let radius = i32(round(params.maskFeather * 6.0));
  if (radius > 0) {
    let maxXY = vec2i(i32(params.resolution.x) - 1, i32(params.resolution.y) - 1);
    var acc = m;
    var w = 1.0;
    let sigmaR = 0.15;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
      for (var dx = -radius; dx <= radius; dx = dx + 1) {
        if (dx == 0 && dy == 0) { continue; }
        let p = clamp(pixel + vec2i(dx, dy), vec2i(0), maxXY);
        let nLuma = luma(textureLoad(frameTex, p, 0).rgb);
        let nUv = (vec2f(p) + 0.5) / res;
        let nMask = textureSampleLevel(rawMaskTex, linearSampler, nUv, 0.0).r;
        let dl = nLuma - centerLuma;
        let weight = exp(-(dl * dl) / (sigmaR * sigmaR));
        acc = acc + nMask * weight;
        w = w + weight;
      }
    }
    m = acc / w;
  }

  // 4. Temporal EMA against previous refined mask
  let prev = textureLoad(prevRefinedTex, pixel, 0).r;
  let final = mix(m, prev, clamp(params.maskStability, 0.0, 0.95));

  textureStore(refinedOut, pixel, vec4f(clamp(final, 0.0, 1.0), 0.0, 0.0, 1.0));
}
```

- [ ] **Step 5.3: Add ping-pong refined mask textures and pipeline to the renderer**

Edit `src/lib/webgpuRenderer.ts`.

Add fields to the class:

```typescript
  private refinedMaskTextures: GPUTexture[] = []
  private refinedMaskIndex = 0
  private maskRefinePipeline: GPUComputePipeline | null = null
  private maskRefineBindGroups: GPUBindGroup[] = []
  private maskUniformBuffer: GPUBuffer | null = null
```

Add the import:

```typescript
import maskRefineShader from './shaders/maskRefine.wgsl?raw'
```

In `init()`, after the existing `this.uniformBuffer = ...` line, create the mask uniform buffer:

```typescript
    this.maskUniformBuffer = this.device.createBuffer({
      size: 8 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'mask-refine-uniforms',
    })
```

In `createTextures()`, append:

```typescript
    this.refinedMaskTextures = [
      this.device.createTexture({ ...base, label: 'refined-mask-a' }),
      this.device.createTexture({ ...base, label: 'refined-mask-b' }),
    ]
```

In `createPipelines()`, before `this.createBindGroups()`, add:

```typescript
    const maskRefineModule = this.device.createShaderModule({ code: maskRefineShader })
    this.maskRefinePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: maskRefineModule, entryPoint: 'main' },
    })
```

In `createBindGroups()`, add (before the existing `computeBindGroups` block):

```typescript
    const refinedViews = this.refinedMaskTextures.map((t) => t.createView())
    const rawMaskView = this.rawMaskTexture!.createView()
    const maskUniformResource = { buffer: this.maskUniformBuffer! }
    this.maskRefineBindGroups = [
      this.device.createBindGroup({
        layout: this.maskRefinePipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: rawMaskView },
          { binding: 1, resource: videoView },
          { binding: 2, resource: refinedViews[0] },
          { binding: 3, resource: refinedViews[1] },
          { binding: 4, resource: maskUniformResource },
          { binding: 5, resource: this.sampler! },
        ],
      }),
      this.device.createBindGroup({
        layout: this.maskRefinePipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: rawMaskView },
          { binding: 1, resource: videoView },
          { binding: 2, resource: refinedViews[1] },
          { binding: 3, resource: refinedViews[0] },
          { binding: 4, resource: maskUniformResource },
          { binding: 5, resource: this.sampler! },
        ],
      }),
    ]
```

In `render()`, after `this.uploadPaint(paintCanvas)` and `this.writeUniforms(...)`, before the `encoder.beginComputePass()`, add:

```typescript
    this.writeMaskUniforms(settings)
```

After the existing `computePass` (motion-based) block, before `const view = context.getCurrentTexture()`, add a new compute pass for mask refine:

```typescript
    const maskPass = encoder.beginComputePass({ label: 'mask-refine' })
    maskPass.setPipeline(this.maskRefinePipeline!)
    maskPass.setBindGroup(0, this.maskRefineBindGroups[this.refinedMaskIndex])
    maskPass.dispatchWorkgroups(
      Math.ceil(RENDER_WIDTH / WORKGROUP_SIZE),
      Math.ceil(RENDER_HEIGHT / WORKGROUP_SIZE),
    )
    maskPass.end()
    this.refinedMaskIndex = 1 - this.refinedMaskIndex
```

Add the helper method:

```typescript
  private writeMaskUniforms(settings: FxSettings): void {
    if (!this.device || !this.maskUniformBuffer) return
    const data = new Float32Array(8)
    data[0] = RENDER_WIDTH
    data[1] = RENDER_HEIGHT
    data[2] = settings.maskFeather
    data[3] = settings.maskStability
    data[4] = settings.followLock
    this.device.queue.writeBuffer(this.maskUniformBuffer, 0, data)
  }
```

In `dispose()`, add:

```typescript
    this.refinedMaskTextures.forEach((t) => t.destroy())
    this.maskUniformBuffer?.destroy()
    this.maskRefineBindGroups = []
```

Update `ready()` to require `this.maskRefinePipeline` and `this.refinedMaskTextures.length === 2`.

- [ ] **Step 5.4: Route the `Matte` debug view to refined mask**

Edit `src/lib/shaders/legacyRender.wgsl`:

Replace the binding declarations (lines starting with `@group(0) @binding(...)`) so the rawMaskTex is replaced by refinedMaskTex. Change:

```wgsl
@group(0) @binding(5) var rawMaskTex: texture_2d<f32>;
```

to:

```wgsl
@group(0) @binding(5) var refinedMaskTex: texture_2d<f32>;
```

In `fragmentMain`, change the matte branch:

```wgsl
  if (params.debugView > 0.5 && params.debugView < 1.5) {
    let m = textureSample(refinedMaskTex, linearSampler, uv).r;
    return vec4f(vec3f(m), 1.0);
  }
```

In `src/lib/webgpuRenderer.ts`, in `createBindGroups()`, replace the `{ binding: 5, resource: rawMaskView }` entry inside the `renderBindGroups = plateViews.map(...)` block with the **refined** mask view of the *opposite* index from current writing target. Easiest: bind both ping-pong refined views into two render bind groups, indexed by `this.refinedMaskIndex`. Replace the render bind groups block with:

```typescript
    this.renderBindGroups = [0, 1].map((plateIdx) =>
      [0, 1].map((maskIdx) =>
        this.device!.createBindGroup({
          layout: this.renderPipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.sampler! },
            { binding: 1, resource: videoView },
            { binding: 2, resource: paintView },
            { binding: 3, resource: plateViews[plateIdx] },
            { binding: 4, resource: uniformResource },
            { binding: 5, resource: refinedViews[maskIdx] },
          ],
        }),
      ),
    ).flat()
```

Note this changes `renderBindGroups` from `GPUBindGroup[]` length 2 to length 4. Update the type and the access in `render()`:

```typescript
    // After mask-refine swap, this.refinedMaskIndex points at the freshly-written refined mask.
    // After legacy compute-pass swap, this.plateIndex points at the freshly-written plate.
    const idx = this.plateIndex * 2 + this.refinedMaskIndex
    renderPass.setBindGroup(0, renderBindGroups[idx])
```

And update `ready()` to check `renderBindGroups.length === 4`.

- [ ] **Step 5.5: Verify**

```bash
npm run lint && npm run build && npm run dev
```

Open Demo. Switch to **Matte** view — should show a refined mask: smoother, edge-aware, slightly dilated. Drag the **Follow lock** slider — the mask should swell and shrink. Final view (default) should still look the same as before (we haven't touched the composite math yet).

If you see a black screen or WebGPU error in the console, the bind group layout likely mismatches the shader — verify binding numbers in `legacyRender.wgsl` match `createBindGroups`.

- [ ] **Step 5.6: Commit**

```bash
git add src/lib/types.ts src/lib/shaders/maskRefine.wgsl src/lib/shaders/legacyRender.wgsl src/lib/webgpuRenderer.ts
git commit -m "feat: add mask refine compute pass with bilateral filter and temporal EMA"
```

---

## Task 6 — Pass B: mask-driven plate learn shader (replace motion-driven)

**Files:**
- Create: `src/lib/shaders/plateLearn.wgsl`
- Modify: `src/lib/webgpuRenderer.ts` (replace the legacy compute pipeline)
- Delete: `src/lib/shaders/legacyCompute.wgsl` (no longer used after this task)

**Steps:**

- [ ] **Step 6.1: Write the plate learn shader**

Create `src/lib/shaders/plateLearn.wgsl`:

```wgsl
struct Params {
  resolution: vec2f,
  plateLearning: f32,
  pad0: f32,
};

@group(0) @binding(0) var frameTex: texture_2d<f32>;
@group(0) @binding(1) var oldPlateTex: texture_2d<f32>;
@group(0) @binding(2) var refinedMaskTex: texture_2d<f32>;
@group(0) @binding(3) var newPlateOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= u32(params.resolution.x) || xy.y >= u32(params.resolution.y)) {
    return;
  }
  let pixel = vec2i(xy);

  let frame = textureLoad(frameTex, pixel, 0);
  let oldPlate = textureLoad(oldPlateTex, pixel, 0);
  let mask = textureLoad(refinedMaskTex, pixel, 0).r;

  let learnGate = clamp(1.0 - mask, 0.0, 1.0);
  let learnRate = clamp(learnGate * params.plateLearning, 0.0, 1.0);
  let nextRgb = mix(oldPlate.rgb, frame.rgb, learnRate);

  let confidenceGain = 0.04;
  let nextConf = clamp(oldPlate.a + learnGate * confidenceGain, 0.0, 1.0);

  textureStore(newPlateOut, pixel, vec4f(nextRgb, nextConf));
}
```

- [ ] **Step 6.2: Replace compute pipeline in renderer**

Edit `src/lib/webgpuRenderer.ts`.

Replace the import:

```typescript
import computeShader from './shaders/legacyCompute.wgsl?raw'
```

with:

```typescript
import plateLearnShader from './shaders/plateLearn.wgsl?raw'
```

Replace the `computeShader` reference inside `createPipelines()`:

```typescript
    const computeModule = this.device.createShaderModule({ code: plateLearnShader })
```

(The variable name `computeModule` and `computePipeline` can stay for less churn — the *contract* changed, not the field names.)

Replace the `computeBindGroups` block in `createBindGroups()` — the new shader binds `frameTex`, `oldPlateTex`, `refinedMaskTex`, `newPlateOut`, `params` (5 bindings, no sampler):

```typescript
    this.computeBindGroups = [
      this.device.createBindGroup({
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: videoView },
          { binding: 1, resource: plateViews[0] },
          { binding: 2, resource: refinedViews[this.refinedMaskIndex] }, // see note
          { binding: 3, resource: plateViews[1] },
          { binding: 4, resource: uniformResource },
        ],
      }),
      this.device.createBindGroup({
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: videoView },
          { binding: 1, resource: plateViews[1] },
          { binding: 2, resource: refinedViews[this.refinedMaskIndex] },
          { binding: 3, resource: plateViews[0] },
          { binding: 4, resource: uniformResource },
        ],
      }),
    ]
```

Note: we need both refined mask views available because plate learn runs *after* mask refine swap. Better: build 4 plate-learn bind groups (2 plate × 2 mask) just like we did for render groups, and pick the right one in `render()`. Replace with:

```typescript
    this.computeBindGroups = [0, 1].flatMap((plateIdx) =>
      [0, 1].map((maskIdx) =>
        this.device!.createBindGroup({
          layout: this.computePipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: videoView },
            { binding: 1, resource: plateViews[plateIdx] },
            { binding: 2, resource: refinedViews[maskIdx] },
            { binding: 3, resource: plateViews[1 - plateIdx] },
            { binding: 4, resource: uniformResource },
          ],
        }),
      ),
    )
```

Length is now 4. Update `ready()` accordingly.

In `render()`, the plate-learn dispatch should happen **after** mask refine swap and **before** render. Replace the existing compute pass section with:

```typescript
    const plateLearnPass = encoder.beginComputePass({ label: 'plate-learn' })
    plateLearnPass.setPipeline(this.computePipeline)
    // refinedMaskIndex (post-swap) = freshly-written refined mask.
    // plateIndex here is the *pre*-swap value: bind group reads plate[plateIndex] and writes plate[1-plateIndex].
    const plateBgIdx = this.plateIndex * 2 + this.refinedMaskIndex
    plateLearnPass.setBindGroup(0, this.computeBindGroups[plateBgIdx])
    plateLearnPass.dispatchWorkgroups(
      Math.ceil(RENDER_WIDTH / WORKGROUP_SIZE),
      Math.ceil(RENDER_HEIGHT / WORKGROUP_SIZE),
    )
    plateLearnPass.end()
    this.plateIndex = 1 - this.plateIndex
```

Replace `this.writeUniforms(settings, time)` body so the legacy uniform fields it wrote are still written (the legacy render shader still reads them) — keep `writeUniforms` exactly as it was for now. Add a separate small uniform buffer for plate learn? No — it only needs `resolution` and `plateLearning`. Reuse a small buffer:

Add field:

```typescript
  private plateUniformBuffer: GPUBuffer | null = null
```

In `init()`:

```typescript
    this.plateUniformBuffer = this.device.createBuffer({
      size: 4 * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'plate-learn-uniforms',
    })
```

In the bind group block above, replace `{ binding: 4, resource: uniformResource }` inside `computeBindGroups` with `{ binding: 4, resource: { buffer: this.plateUniformBuffer! } }`.

Add helper:

```typescript
  private writePlateUniforms(settings: FxSettings): void {
    if (!this.device || !this.plateUniformBuffer) return
    const data = new Float32Array(4)
    data[0] = RENDER_WIDTH
    data[1] = RENDER_HEIGHT
    data[2] = settings.plateLearning
    data[3] = 0
    this.device.queue.writeBuffer(this.plateUniformBuffer, 0, data)
  }
```

Call it in `render()` alongside `writeMaskUniforms`.

In `dispose()`:

```typescript
    this.plateUniformBuffer?.destroy()
```

Note: the plate texture format was previously RGBA8 with the alpha channel unused. We're now writing confidence into alpha. The existing texture descriptor (`format: 'rgba8unorm'`) is fine. The `Reset plate` operation already does `copyTextureToTexture(video → plate)` which copies all 4 channels — since the camera frame has alpha=255 (1.0), confidence will start at 1.0 right after reset. That's actually wrong — after reset, confidence should be 0 (we haven't observed the background yet, we just snapshot the current frame as a guess). Fix: after reset, replace the copyTextureToTexture with a clear or with a write of `(frame.rgb, 0)`. Easiest: after the existing copy, run a one-shot dispatch that zeros the alpha channel. Or, simpler: change the initial copy to write `(frame.rgb, 0.5)` — initial guess with medium confidence. Cleanest: skip the copy entirely; let plate accumulate from black. But that means frames 1–30 show black background where mask is. Compromise: do the copy AND set confidence to a small value like 0.3 so push-pull starts taking over after a few frames if user is in shot.

For simplicity in this task, leave the existing `copyTextureToTexture` (sets alpha to 1.0); push-pull won't kick in until user is out of frame and confidence drops elsewhere. We'll revisit in Task 12 if behavior is wrong.

- [ ] **Step 6.3: Delete the legacy compute shader file**

```bash
rm src/lib/shaders/legacyCompute.wgsl
```

- [ ] **Step 6.4: Verify**

```bash
npm run lint && npm run build && npm run dev
```

Open Demo. Switch debug view to **Plate** — should show a slowly-clearing background image. Walk in front of the camera (or use Demo) — the silhouette should NOT get learned into the plate (this is the key test). After a few seconds with the silhouette moving around, click **Reset plate** — plate should snap to the current frame.

Switch debug view to **Matte** — refined mask should still look correct.

Switch back to **Final** — composite still uses legacy render math, so the result will look weird/wrong (the legacy render uses old motion math but the plate is now mask-driven). That's expected; we fix it in Task 8.

- [ ] **Step 6.5: Commit**

```bash
git add src/lib/shaders/plateLearn.wgsl src/lib/webgpuRenderer.ts
git rm src/lib/shaders/legacyCompute.wgsl
git commit -m "feat: replace motion-driven plate learn with ML mask-driven, add confidence channel"
```

---

## Task 7 — Pass C: push-pull pyramid inpainting

**Files:**
- Create: `src/lib/shaders/pushDown.wgsl`
- Create: `src/lib/shaders/pushUp.wgsl`
- Modify: `src/lib/webgpuRenderer.ts` (add pyramid textures, two pipelines, dispatch chains)

**Steps:**

- [ ] **Step 7.1: Write `pushDown.wgsl`**

```wgsl
struct DstSize { size: vec2u, };

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var dstOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> info: DstSize;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= info.size.x || xy.y >= info.size.y) { return; }
  let p = vec2i(xy) * 2;
  var acc = vec3f(0.0);
  var w = 0.0;
  for (var dy = 0; dy < 2; dy = dy + 1) {
    for (var dx = 0; dx < 2; dx = dx + 1) {
      let s = textureLoad(srcTex, p + vec2i(dx, dy), 0);
      acc = acc + s.rgb * s.a;
      w = w + s.a;
    }
  }
  let rgb = select(vec3f(0.0), acc / max(w, 0.0001), w > 0.001);
  let outA = clamp(w * 0.25, 0.0, 1.0);
  textureStore(dstOut, vec2i(xy), vec4f(rgb, outA));
}
```

- [ ] **Step 7.2: Write `pushUp.wgsl`**

```wgsl
struct DstSize { size: vec2u, };

@group(0) @binding(0) var sameLevelTex: texture_2d<f32>;
@group(0) @binding(1) var coarseLevelTex: texture_2d<f32>;
@group(0) @binding(2) var dstOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var coarseSampler: sampler;
@group(0) @binding(4) var<uniform> info: DstSize;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= info.size.x || xy.y >= info.size.y) { return; }
  let pixel = vec2i(xy);
  let same = textureLoad(sameLevelTex, pixel, 0);
  let uv = (vec2f(xy) + 0.5) / vec2f(info.size);
  let coarse = textureSampleLevel(coarseLevelTex, coarseSampler, uv, 0.0);
  let trust = smoothstep(0.0, 0.95, same.a);
  let rgb = mix(coarse.rgb, same.rgb, trust);
  let a = max(same.a, coarse.a * 0.9);
  textureStore(dstOut, pixel, vec4f(rgb, a));
}
```

- [ ] **Step 7.3: Add pyramid textures and pipelines to renderer**

Edit `src/lib/webgpuRenderer.ts`.

Add imports:

```typescript
import pushDownShader from './shaders/pushDown.wgsl?raw'
import pushUpShader from './shaders/pushUp.wgsl?raw'
```

Add fields:

```typescript
  private pyramidDown: GPUTexture[] = [] // length 6: levels 0..5, level 0 is full-res copy of plate
  private pyramidUp: GPUTexture[] = []   // length 6: levels 0..5
  private pushDownPipeline: GPUComputePipeline | null = null
  private pushUpPipeline: GPUComputePipeline | null = null
  private pushDownBindGroups: GPUBindGroup[] = [] // length 5 (one per down step)
  private pushUpBindGroups: GPUBindGroup[] = []   // length 5 (one per up step)
  private pyramidSizeUniforms: GPUBuffer[] = []   // length 6 (one per level)
```

Define pyramid sizes — add a top-level constant:

```typescript
const PYRAMID_SIZES: Array<[number, number]> = [
  [RENDER_WIDTH, RENDER_HEIGHT], // 960×540
  [480, 270],
  [240, 135],
  [120, 68],
  [60, 34],
  [30, 17],
]
```

In `createTextures()`, after the refined mask textures, add:

```typescript
    this.pyramidDown = PYRAMID_SIZES.map(([w, h], i) =>
      this.device!.createTexture({
        size: { width: w, height: h },
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.COPY_DST,
        label: `pyramid-down-${i}`,
      }),
    )
    this.pyramidUp = PYRAMID_SIZES.map(([w, h], i) =>
      this.device!.createTexture({
        size: { width: w, height: h },
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.COPY_SRC,
        label: `pyramid-up-${i}`,
      }),
    )
```

In `init()`, after creating `plateUniformBuffer`, create the per-level uniform buffers:

```typescript
    this.pyramidSizeUniforms = PYRAMID_SIZES.map(([w, h], i) => {
      const buffer = this.device!.createBuffer({
        size: 4 * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: `pyramid-size-${i}`,
      })
      this.device!.queue.writeBuffer(buffer, 0, new Uint32Array([w, h, 0, 0]))
      return buffer
    })
```

In `createPipelines()`, after the mask refine pipeline:

```typescript
    const pushDownModule = this.device.createShaderModule({ code: pushDownShader })
    this.pushDownPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: pushDownModule, entryPoint: 'main' },
    })
    const pushUpModule = this.device.createShaderModule({ code: pushUpShader })
    this.pushUpPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: pushUpModule, entryPoint: 'main' },
    })
```

In `createBindGroups()`, append:

```typescript
    const downViews = this.pyramidDown.map((t) => t.createView())
    const upViews = this.pyramidUp.map((t) => t.createView())
    this.pushDownBindGroups = []
    for (let i = 0; i < 5; i++) {
      this.pushDownBindGroups.push(
        this.device.createBindGroup({
          layout: this.pushDownPipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: downViews[i] },
            { binding: 1, resource: downViews[i + 1] },
            { binding: 2, resource: { buffer: this.pyramidSizeUniforms[i + 1] } },
          ],
        }),
      )
    }
    this.pushUpBindGroups = []
    for (let i = 0; i < 5; i++) {
      this.pushUpBindGroups.push(
        this.device.createBindGroup({
          layout: this.pushUpPipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: downViews[i] },
            { binding: 1, resource: upViews[i + 1] },
            { binding: 2, resource: upViews[i] },
            { binding: 3, resource: this.sampler! },
            { binding: 4, resource: { buffer: this.pyramidSizeUniforms[i] } },
          ],
        }),
      )
    }
```

In `render()`, after the plate learn pass and the `this.plateIndex` swap, before the render pass:

```typescript
    // Push-pull: copy current plate into pyramidDown[0]
    encoder.copyTextureToTexture(
      { texture: this.plateTextures[1 - this.plateIndex] }, // freshly written plate (the new one, since we already swapped)
      { texture: this.pyramidDown[0] },
      { width: RENDER_WIDTH, height: RENDER_HEIGHT },
    )
    // Down chain
    for (let i = 0; i < 5; i++) {
      const pass = encoder.beginComputePass({ label: `push-down-${i}` })
      pass.setPipeline(this.pushDownPipeline!)
      pass.setBindGroup(0, this.pushDownBindGroups[i])
      const [w, h] = PYRAMID_SIZES[i + 1]
      pass.dispatchWorkgroups(Math.ceil(w / WORKGROUP_SIZE), Math.ceil(h / WORKGROUP_SIZE))
      pass.end()
    }
    // Initialize up[5] = down[5]
    encoder.copyTextureToTexture(
      { texture: this.pyramidDown[5] },
      { texture: this.pyramidUp[5] },
      { width: PYRAMID_SIZES[5][0], height: PYRAMID_SIZES[5][1] },
    )
    // Up chain
    for (let i = 4; i >= 0; i--) {
      const pass = encoder.beginComputePass({ label: `push-up-${i}` })
      pass.setPipeline(this.pushUpPipeline!)
      pass.setBindGroup(0, this.pushUpBindGroups[i])
      const [w, h] = PYRAMID_SIZES[i]
      pass.dispatchWorkgroups(Math.ceil(w / WORKGROUP_SIZE), Math.ceil(h / WORKGROUP_SIZE))
      pass.end()
    }
```

Wait — `this.plateTextures[1 - this.plateIndex]` after the swap: we just did `this.plateIndex = 1 - this.plateIndex`. Before swap, freshly-written index was `1 - originalIndex`. After swap, `this.plateIndex == 1 - originalIndex`, so freshly-written is at `this.plateIndex`. Correct line:

```typescript
      { texture: this.plateTextures[this.plateIndex] }, // freshly written plate after swap
```

Use that. Update accordingly.

In `dispose()`:

```typescript
    this.pyramidDown.forEach((t) => t.destroy())
    this.pyramidUp.forEach((t) => t.destroy())
    this.pyramidSizeUniforms.forEach((b) => b.destroy())
```

- [ ] **Step 7.4: Verify**

```bash
npm run lint && npm run build && npm run dev
```

Open Demo. WebGPU shouldn't error. The Final view will still look broken (composite still legacy). To validate push-pull is producing something, temporarily peek `pyramidUp[0]` — easiest way is to extend the legacy render shader's `paint` debug view to sample it, but that requires a binding change. **Skip explicit visual check this task — trust the dispatch will be validated when Task 8's composite uses it.**

Console should be clean (no WebGPU validation errors). If you see `Buffer is too small` or `Texture binding format mismatch`, fix the binding layout before continuing.

- [ ] **Step 7.5: Commit**

```bash
git add src/lib/shaders/pushDown.wgsl src/lib/shaders/pushUp.wgsl src/lib/webgpuRenderer.ts
git commit -m "feat: add push-pull pyramid inpainting (5 down + 5 up dispatches)"
```

---

## Task 8 — Pass D: new composite shader replacing legacy render

**Files:**
- Create: `src/lib/shaders/composite.wgsl`
- Modify: `src/lib/webgpuRenderer.ts` (replace render pipeline + bind groups)
- Modify: `src/lib/types.ts` (add `inpaintFallback` field, change `DebugView` enum)
- Delete: `src/lib/shaders/legacyRender.wgsl` (no longer used)

**Steps:**

- [ ] **Step 8.1: Add `inpaintFallback` and update `DebugView`**

Edit `src/lib/types.ts`:

Change `DebugView`:

```typescript
export type DebugView = 'final' | 'matte' | 'plate' | 'inpaint'
```

In `FxSettings`, add:

```typescript
  inpaintFallback: number
```

In `DEFAULT_SETTINGS`, add:

```typescript
  inpaintFallback: 0.85,
```

Remove `paintPreview` from `FxSettings` and from `DEFAULT_SETTINGS`. Remove `brushSize`, `brushMode` too — App.tsx still references them (we'll fix in Task 11), so leave them for now to keep the build green. **Defer paint field removal to Task 11.**

Actually, change of plan: remove `paintPreview` only (it's used by the legacy render shader we're deleting). Keep `brushSize` and `brushMode` for now.

Wait — `paintPreview` is also referenced in `App.tsx` (line ~427). To avoid a broken build between tasks, leave `paintPreview` in `FxSettings` until Task 11. Just add `inpaintFallback` and change `DebugView`.

In `App.tsx`, the `<select>` for debug view (around line 444) references `'paint'` as an option. Change that option's value/label to `'inpaint'` / `Inpaint`:

```typescript
              <option value="final">Final</option>
              <option value="matte">Matte</option>
              <option value="plate">Plate</option>
              <option value="inpaint">Inpaint</option>
```

- [ ] **Step 8.2: Write the composite shader**

Create `src/lib/shaders/composite.wgsl`:

```wgsl
struct Params {
  resolution: vec2f,
  time: f32,
  opacity: f32,
  jelly: f32,
  water: f32,
  cloth: f32,
  refraction: f32,
  edgeGain: f32,
  inpaintFallback: f32,
  debugView: f32,
  pad0: f32, pad1: f32, pad2: f32, pad3: f32, pad4: f32,
};

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var frameTex: texture_2d<f32>;
@group(0) @binding(2) var plateTex: texture_2d<f32>;
@group(0) @binding(3) var inpaintTex: texture_2d<f32>;
@group(0) @binding(4) var refinedMaskTex: texture_2d<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  let pos = positions[index];
  var out: VertexOut;
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = vec2f((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
  return out;
}

@fragment
fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  let uv = input.uv;
  let mask = textureSample(refinedMaskTex, linearSampler, uv).r;

  if (params.debugView > 0.5 && params.debugView < 1.5) {
    return vec4f(vec3f(mask), 1.0);
  }
  if (params.debugView > 1.5 && params.debugView < 2.5) {
    return vec4f(textureSample(plateTex, linearSampler, uv).rgb, 1.0);
  }
  if (params.debugView > 2.5 && params.debugView < 3.5) {
    return vec4f(textureSample(inpaintTex, linearSampler, uv).rgb, 1.0);
  }

  let t = params.time;
  let aspect = params.resolution.x / params.resolution.y;
  let centered = vec2f((uv.x - 0.5) * aspect, uv.y - 0.5);

  let jellyOffset = vec2f(
    sin((uv.y * 24.0) + t * 4.0) * 0.012,
    sin((uv.x * 18.0) - t * 2.7) * 0.007,
  ) * params.jelly;
  let waterOffset = vec2f(
    sin((centered.x * 40.0) + t * 5.2) + sin((uv.y * 65.0) - t * 3.4),
    cos((centered.y * 38.0) - t * 4.6),
  ) * 0.006 * params.water;
  let clothOffset = vec2f(
    sin((uv.y * 92.0) + t * 1.6) * 0.003,
    cos((uv.x * 76.0) - t * 1.2) * 0.003,
  ) * params.cloth;
  let displacement = (jellyOffset + waterOffset + clothOffset) * params.refraction * mask;

  let plateSamp = textureSample(plateTex, linearSampler, uv + displacement);
  let inpaintSamp = textureSample(inpaintTex, linearSampler, uv + displacement);
  let fillBlend = clamp(params.inpaintFallback * (1.0 - plateSamp.a), 0.0, 1.0);
  let bg = mix(plateSamp.rgb, inpaintSamp.rgb, fillBlend);

  let caustic = (sin((uv.x + uv.y) * 80.0 + t * 5.0) * 0.5 + 0.5) * params.water;
  let weave = (sin(uv.x * 180.0) * sin(uv.y * 150.0) * 0.5 + 0.5) * params.cloth;
  let materialLight = mask * (caustic * 0.1 + weave * 0.055 + params.jelly * 0.045);

  let frame = textureSample(frameTex, linearSampler, uv).rgb;
  let reconstructed = bg + materialLight;
  let composed = mix(frame, reconstructed, mask * params.opacity);

  let edge = smoothstep(0.12, 0.48, mask) * (1.0 - smoothstep(0.52, 0.9, mask));
  let edgeColor = vec3f(0.52, 0.94, 1.0) * edge * params.edgeGain;

  return vec4f(composed + edgeColor, 1.0);
}
```

- [ ] **Step 8.3: Replace render pipeline in renderer**

Edit `src/lib/webgpuRenderer.ts`.

Replace the import:

```typescript
import renderShader from './shaders/legacyRender.wgsl?raw'
```

with:

```typescript
import compositeShader from './shaders/composite.wgsl?raw'
```

Replace the `renderShader` reference inside `createPipelines()`:

```typescript
    const renderModule = this.device.createShaderModule({ code: compositeShader })
```

Replace the `renderBindGroups` block in `createBindGroups()` (currently 4-element with paint texture). New layout: sampler, frameTex, plateTex, inpaintTex, refinedMaskTex, params (6 bindings). 4 combinations of (plateIdx × maskIdx). Note plate and inpaint are independent (inpaint always reads `pyramidUp[0]`):

```typescript
    const inpaintView = this.pyramidUp[0].createView()
    this.renderBindGroups = [0, 1].flatMap((plateIdx) =>
      [0, 1].map((maskIdx) =>
        this.device!.createBindGroup({
          layout: this.renderPipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.sampler! },
            { binding: 1, resource: videoView },
            { binding: 2, resource: plateViews[plateIdx] },
            { binding: 3, resource: inpaintView },
            { binding: 4, resource: refinedViews[maskIdx] },
            { binding: 5, resource: uniformResource },
          ],
        }),
      ),
    )
```

In `render()`, the render pass setBindGroup needs the freshly-written plate and freshly-written mask. After both ping-pong swaps, `this.plateIndex` and `this.refinedMaskIndex` already point at the freshly-written textures:

```typescript
    const freshPlateIdx = this.plateIndex
    const freshMaskIdxRender = this.refinedMaskIndex
    renderPass.setBindGroup(0, renderBindGroups[freshPlateIdx * 2 + freshMaskIdxRender])
```

Update `writeUniforms()` to write the new composite uniform layout (16 floats matching the `Params` struct):

```typescript
  private writeUniforms(settings: FxSettings, time: number): void {
    if (!this.device || !this.uniformBuffer) return
    const data = new Float32Array(16)
    data[0] = RENDER_WIDTH
    data[1] = RENDER_HEIGHT
    data[2] = time * 0.001
    data[3] = settings.opacity
    data[4] = settings.jelly
    data[5] = settings.water
    data[6] = settings.cloth
    data[7] = settings.refraction
    data[8] = settings.edgeGain
    data[9] = settings.inpaintFallback
    data[10] = debugViewToNumber(settings.debugView)
    // 11..15 padding
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data)
  }
```

Update `debugViewToNumber`:

```typescript
function debugViewToNumber(view: DebugView): number {
  if (view === 'matte') return 1
  if (view === 'plate') return 2
  if (view === 'inpaint') return 3
  return 0
}
```

Remove `paintTexture`, `uploadPaint`, `uploadEmptyPaint`, `emptyPaint`, and the `paintCanvas` parameter from `render()`. Update the new `render()` signature:

```typescript
  render(
    source: HTMLCanvasElement,
    maskSource: OffscreenCanvas | HTMLCanvasElement | null,
    settings: FxSettings,
    time: number,
  ): void {
```

Remove the corresponding uploadPaint call inside.

In `dispose()`, remove `this.paintTexture?.destroy()`.
In `createTextures()`, remove the `paintTexture` creation.
In `createBindGroups()`, remove `paintView`.
In `init()`, remove `this.uploadEmptyPaint()`.

In `App.tsx`, update the `renderer.render(...)` call (around line 89):

```typescript
    renderer.render(frameCanvas, maskCanvas, settingsRef.current, time)
```

(We'll do the full paint UI removal in Task 11 — just unblock the build here.)

- [ ] **Step 8.4: Delete the legacy render shader file**

```bash
rm src/lib/shaders/legacyRender.wgsl
```

- [ ] **Step 8.5: Verify**

```bash
npm run lint && npm run build && npm run dev
```

Open Demo. Default **Final** view should now show real invisibility: the moving silhouette is replaced with the learned background, push-pull fills the holes, edges have a subtle blue glow.

Switch through debug views:
- **Matte** — clean refined mask
- **Plate** — slowly-clearing background, alpha hidden but RGB visible
- **Inpaint** — fully-filled background (push-pull output, no holes)

Switch to Camera. Walk in/out of frame. Stand still — you should remain invisible (key win). Click **Reset plate** when out of frame for a clean baseline. Drag every slider — they should all do something visible.

**Known acceptable issue at this stage:** the FX section panel still has Jelly/Water/Cloth/Refraction (good) but App.tsx may still render the Paint Mask section (fine, it just doesn't drive anything anymore). Cleanup is Task 11.

- [ ] **Step 8.6: Commit**

```bash
git add src/lib/types.ts src/lib/shaders/composite.wgsl src/lib/webgpuRenderer.ts src/App.tsx
git rm src/lib/shaders/legacyRender.wgsl
git commit -m "feat: replace legacy render with ML-mask composite (plate + push-pull + FX)"
```

---

## Task 9 — Add the new sliders to the UI (`maskFeather`, `maskStability`, `inpaintFallback`)

**Files:**
- Modify: `src/App.tsx` (Compositor section)

**Steps:**

- [ ] **Step 9.1: Add three sliders to the Compositor panel**

Edit `src/App.tsx`. In the `<section className="panel-section">` for Compositor (around line 276), after the existing `Plate learning` slider and before the `Reset plate` button, add:

```typescript
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
```

- [ ] **Step 9.2: Verify**

```bash
npm run lint && npm run build && npm run dev
```

Open the panel; the three new sliders appear. Drag each one:
- **Edge feather** at 0 → mask edges crisp/jagged. At 1 → noticeably softer.
- **Mask stability** at 0 → mask responds instantly, possibly jittery. At 0.95 → mask lags noticeably during fast motion.
- **Inpaint fallback** at 0 → background reveals the original frame where plate is unconfident (ugly during early frames). At 1 → push-pull fully takes over for low-confidence regions (smooth).

- [ ] **Step 9.3: Commit**

```bash
git add src/App.tsx
git commit -m "feat: add Edge feather / Mask stability / Inpaint fallback sliders"
```

---

## Task 10 — Repurpose `Follow lock` semantics (UI label only)

**Files:**
- Modify: `src/App.tsx` (relabel slider, optionally update tooltip text)

**Steps:**

- [ ] **Step 10.1: Adjust the Follow lock slider label**

The `followLock` field is now used by `maskRefine.wgsl` as the dilation radius for the ML mask. Update the slider label so users understand. In `src/App.tsx`, find the `Follow lock` SliderControl and change its `label` prop:

```typescript
          <SliderControl
            label="Mask coverage"
            value={settings.followLock}
            ...
          />
```

(Keep the `followLock` field name to avoid type churn; only the label changes.)

- [ ] **Step 10.2: Verify**

```bash
npm run lint && npm run build
```

- [ ] **Step 10.3: Commit**

```bash
git add src/App.tsx
git commit -m "feat: relabel Follow lock as Mask coverage to reflect new ML mask semantics"
```

---

## Task 11 — Remove paint mode (UI, state, types, renderer texture)

**Files:**
- Modify: `src/lib/types.ts` (delete `mode`, `brushSize`, `brushMode`, `paintPreview` from `FxSettings`)
- Modify: `src/App.tsx` (delete paint canvas, pointer handlers, paint UI sections, `paint` mode logic)
- Modify: `src/App.css` (remove `.paint-canvas` styles, segmented `.is-painting`, etc.)
- Modify: `src/lib/webgpuRenderer.ts` (already largely removed in Task 8 — verify clean)

**Steps:**

- [ ] **Step 11.1: Strip paint fields from `FxSettings`**

Edit `src/lib/types.ts`:

Remove `InvisibilityMode`, `BrushMode` types. Remove from `FxSettings`: `mode`, `brushSize`, `brushMode`, `paintPreview`. Remove from `DEFAULT_SETTINGS` the same keys.

Final `FxSettings`:

```typescript
export type DebugView = 'final' | 'matte' | 'plate' | 'inpaint'

export type FxSettings = {
  opacity: number
  matteLow: number
  matteHigh: number
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
  matteLow: 0.24,
  matteHigh: 0.62,
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
```

`matteLow` and `matteHigh` are no longer read by any shader (the mask comes from ML now), but several App.tsx sliders reference them. Remove the sliders along with the fields. Final cleanup: also delete `matteLow` and `matteHigh` from FxSettings + DEFAULT_SETTINGS.

Updated final FxSettings (no matteLow/matteHigh):

```typescript
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
```

- [ ] **Step 11.2: Strip paint UI from `App.tsx`**

In `src/App.tsx`:

- Delete the imports `Eraser`, `Paintbrush`, `Pause` if Pause is unused (Pause is used by Stop button — keep it). Delete `Eraser`, `Paintbrush`.
- Remove `paintCanvasRef` ref and the paint canvas `<canvas>` element (the one with `paint-canvas` className).
- Remove pointer handler functions: `handlePointerDown`, `handlePointerMove`, `handlePointerUp`, `getCanvasPoint`, `drawPaintStroke`. Remove `pointerDownRef`, `lastPointRef`.
- In `useEffect` that initializes canvas sizes (around line 102), delete the `paintCanvas` references and the `paintCanvas.getContext('2d')?.clearRect(...)` line.
- In `renderFrame`, remove `paintCanvas` references; the `renderer.render` call should already be 4-arg `(frameCanvas, maskCanvas, settings, time)` from Task 8.
- Remove the entire **Paint Mask** `<section className="panel-section">` (around lines 391–435).
- Remove the segmented control for "Full body / Paint" mode (around lines 281–296). Replace with nothing — there is only one mode now.
- Remove the `coreCopy` `useMemo` mode-dependent string. Replace with a static line: `'Snapchat-class invisibility — ML person mask + temporal background plate + push-pull inpainting.'`
- Remove the `clearPaint` callback.
- Remove all references to `settings.mode`, `settings.brushMode`, `settings.brushSize`, `settings.paintPreview`.

- [ ] **Step 11.3: Strip paint CSS**

Edit `src/App.css`. Delete CSS rules targeting `.paint-canvas`, `.paint-canvas.is-painting`, and any `.icon-row` / `.field` rules that were ONLY used by the paint section (likely none — leave shared rules).

If unsure, check by searching:

```bash
grep -n "paint-canvas\|is-painting" src/App.css
```

Delete the matching rule blocks.

- [ ] **Step 11.4: Verify**

```bash
npm run lint && npm run build && npm run dev
```

Browser:
- The mode toggle is gone.
- The Paint Mask section is gone.
- Stage shows ONLY the FX canvas; pointer interactions on it should do nothing.
- All other sliders work; the effect renders correctly.

- [ ] **Step 11.5: Commit**

```bash
git add src/lib/types.ts src/App.tsx src/App.css
git commit -m "refactor: remove paint mode (UI, state, fields, CSS)"
```

---

## Task 12 — Plate confidence post-reset cleanup

**Why this exists:** In Task 6 we left a footnote — after `Reset plate`, the plate's alpha (confidence) is set to 1.0 because we copied the camera frame whole. If the user is in front of the camera at reset time, that frame contains them, and confidence will be high everywhere → push-pull never kicks in even though plate is wrong. Fix: zero confidence on reset, let it accumulate from there.

**Files:**
- Modify: `src/lib/webgpuRenderer.ts` (`resetPlate` and the first-frame copy block)

**Steps:**

- [ ] **Step 12.1: Add a confidence-clear pass on reset**

Currently, in `render()`, when `!this.plateInitialized`, the code does `copyTextureToTexture(videoTexture → plate)` for both plates. Change this so RGB starts from the camera but confidence starts at 0.

The simplest correct approach: don't copy at all. Initialize plate textures cleared to (0,0,0,0). Push-pull will fill the screen from neighbors during the first few frames (mostly black at edges → black background blooming inward). That looks bad. Better: copy frame to RGB but explicitly clear alpha.

Option A: clear plate to opaque-black via `clearValue` on a temporary render pass — overkill.
Option B: write a tiny "init plate" compute shader.
Option C: write zeros via `device.queue.writeTexture` — cheap.

Go with C. Replace the `if (!this.plateInitialized)` block in `render()`:

```typescript
    if (!this.plateInitialized) {
      // Seed RGB with current frame, set confidence to 0.
      for (const plate of this.plateTextures) {
        encoder.copyTextureToTexture(
          { texture: videoTexture },
          { texture: plate },
          { width: RENDER_WIDTH, height: RENDER_HEIGHT },
        )
      }
      // Clear the alpha channel by overwriting with a single-row clear
      // (alpha-only writes via copyTextureToTexture aren't possible; use writeTexture full clear).
      const zeros = new Uint8Array(RENDER_WIDTH * RENDER_HEIGHT * 4)
      // Copy frame's RGB but with alpha=0 — but we don't have frame data on CPU.
      // Compromise: leave RGB as-copied, run a follow-up dispatch to zero alpha.
      this.plateInitialized = true
      this.plateNeedsConfidenceReset = true
    }
```

That comment block shows the issue. Switch to a dedicated tiny shader.

Better implementation — add `plateClearConfidencePipeline`:

Create `src/lib/shaders/plateClearConfidence.wgsl`:

```wgsl
struct Params { resolution: vec2f, };

@group(0) @binding(0) var srcRgbTex: texture_2d<f32>;
@group(0) @binding(1) var dstOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= u32(params.resolution.x) || xy.y >= u32(params.resolution.y)) { return; }
  let pixel = vec2i(xy);
  let rgb = textureLoad(srcRgbTex, pixel, 0).rgb;
  textureStore(dstOut, pixel, vec4f(rgb, 0.0));
}
```

In `webgpuRenderer.ts`:

Add the import:

```typescript
import plateClearConfidenceShader from './shaders/plateClearConfidence.wgsl?raw'
```

Add fields:

```typescript
  private plateClearPipeline: GPUComputePipeline | null = null
  private plateClearBindGroups: GPUBindGroup[] = []
```

In `createPipelines()`:

```typescript
    const plateClearModule = this.device.createShaderModule({ code: plateClearConfidenceShader })
    this.plateClearPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: plateClearModule, entryPoint: 'main' },
    })
```

In `createBindGroups()`:

```typescript
    this.plateClearBindGroups = [0, 1].map((idx) =>
      this.device!.createBindGroup({
        layout: this.plateClearPipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: videoView }, // src is current frame
          { binding: 1, resource: plateViews[idx] },
          { binding: 2, resource: { buffer: this.plateUniformBuffer! } }, // reuse, only reads .xy
        ],
      }),
    )
```

In `render()`, replace the existing init block:

```typescript
    if (!this.plateInitialized) {
      for (const bindGroup of this.plateClearBindGroups) {
        const initPass = encoder.beginComputePass({ label: 'plate-init' })
        initPass.setPipeline(this.plateClearPipeline!)
        initPass.setBindGroup(0, bindGroup)
        initPass.dispatchWorkgroups(
          Math.ceil(RENDER_WIDTH / WORKGROUP_SIZE),
          Math.ceil(RENDER_HEIGHT / WORKGROUP_SIZE),
        )
        initPass.end()
      }
      this.plateInitialized = true
    }
```

- [ ] **Step 12.2: Verify**

```bash
npm run lint && npm run build && npm run dev
```

Camera in your face. Click **Reset plate**. You should see a brief flash where the inpaint fallback fills the screen (push-pull blurring), then plate confidence rebuilds wherever you're not. Move out of frame for ~2 seconds — plate should fully clear up.

- [ ] **Step 12.3: Commit**

```bash
git add src/lib/shaders/plateClearConfidence.wgsl src/lib/webgpuRenderer.ts
git commit -m "fix: zero plate confidence on reset so push-pull takes over until background is observed"
```

---

## Task 13 — Update `README.md`

**Files:**
- Modify: `README.md`

**Steps:**

- [ ] **Step 13.1: Rewrite the relevant sections**

Replace the existing README content with:

```markdown
# InvisibleCam

Open-source browser webcam invisibility filter for MacBook-class hardware. Snapchat / Effect House class effect, fully in-browser.

## Pipeline

InvisibleCam uses a WebGPU compute + render pipeline driven by an in-browser ML person segmenter:

1. **Segmentation** — `@mediapipe/tasks-vision` ImageSegmenter (`selfie-multiclass-256x256`, GPU delegate) produces a 256×256 person confidence mask.
2. **Mask refine** — WGSL compute pass: bilinear upsample, dilation, joint-bilateral edge snap, temporal EMA.
3. **Plate learn** — temporally accumulated background plate (RGBA8, alpha = observation confidence). Mask-gated learning means the foreground subject never contaminates the plate.
4. **Push-pull inpainting** — 5-level pyramid fills regions the plate has not yet observed.
5. **Composite** — full-screen fragment shader blends frame + plate + inpaint with optional jelly / water / cloth refraction overlays.

Total budget on M1/M2 MacBooks: ~10–15 ms per 960×540 frame at 60 fps.

## Run

```bash
npm install
npm run dev
```

Open the Vite localhost URL in a WebGPU-capable Chromium browser. Camera capture requires localhost or HTTPS. The MediaPipe model and WASM bundle are fetched from the MediaPipe CDN on first load (~3 MB total).

## Verification

```bash
npm run lint
npm run build
```
```

- [ ] **Step 13.2: Verify**

```bash
cat README.md  # eyeball the result
```

(Skip `npm` commands — README change cannot break anything.)

- [ ] **Step 13.3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for the ML-driven pipeline"
```

---

## Task 14 — End-to-end smoke test (spec §8)

**Files:** none modified.

**Steps:**

- [ ] **Step 14.1: Final lint and build**

```bash
npm run lint && npm run build
```

Both must pass with zero errors and zero warnings. If anything new shows up, fix it before continuing.

- [ ] **Step 14.2: Manual checklist in browser**

```bash
npm run dev
```

Open the localhost URL in a WebGPU-capable Chromium browser. Verify each of the following — every item must pass:

1. Page loads, status pill says **Renderer: <GPU name>** (not error).
2. Default mode is **Demo**. The animated silhouette in the demo source is replaced with the reconstructed background. FPS pill shows ≥ 50.
3. Click **Camera**, allow permission. Switch debug view → **Matte**: a clean white silhouette of you appears.
4. Switch back to **Final**. Walk into frame from off-screen → your body is invisible from the moment you appear (no need for prior background capture).
5. Stand completely still for 5 seconds. **You stay invisible.** (This is the regression check vs the old algorithm.)
6. Turn your head sharply left and right. Mask follows without leaving persistent ghost trails.
7. Drag every slider in every panel section. Each one produces a visible change in the **Final** view (no dead controls).
8. Click **Reset plate** while in frame. There's a momentary flicker (push-pull bloom), then plate rebuilds.
9. Cycle through all four debug views: **Final**, **Matte**, **Plate**, **Inpaint**. Each renders something coherent.
10. FPS pill stays ≥ 50 during all of the above.

If any item fails, file a bug back to the implementer with which step + what you saw vs expected. Do NOT mark this task complete until all 10 pass.

- [ ] **Step 14.3: Commit (if any fixes were needed during smoke test)**

If you made fix commits, no final commit needed. If everything passed first try, no commit.

---

## Done

Total touchpoints: 7 new shaders, 1 new TypeScript module, 5 modified files, 2 deleted files. Estimated implementer time: 4–6 hours.
