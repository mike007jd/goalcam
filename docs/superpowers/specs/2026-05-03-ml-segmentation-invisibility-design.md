# ML-Driven Invisibility — Design Spec

**Date**: 2026-05-03
**Status**: Approved (pending implementation plan)
**Owner**: InvisibleCam

---

## 1. Goal

Replace the current motion-difference-based invisibility algorithm with a Snapchat / Effect House class effect: detect the person via ML segmentation, then reconstruct the occluded background using a combination of a temporally learned plate and a shader-side push-pull inpainting fallback. Real-time on MacBook-class hardware, in-browser, WebGPU end-to-end after the segmentation step.

The current algorithm fails the moment the user stops moving (they get learned into the plate) or moves into a region the plate has not seen (they leak through). It also has rough mask edges. ML segmentation eliminates both class of failure at the source.

## 2. Scope

### In scope
- Add a browser-side person segmentation model (MediaPipe Tasks Vision — ImageSegmenter, `selfie-multiclass-256x256`).
- Replace the current motion-based mask compute pass with an ML-mask-driven plate-learning pass.
- Add a multi-pass mask refinement step (upsample, guided filter, morphological close, temporal EMA).
- Add a push-pull pyramid inpainting pass to fill plate-confidence holes.
- Refactor `webgpuRenderer.ts` into a multi-pass orchestrator with shader sources extracted into separate `.wgsl` files.
- Remove the paint mode (UI, mask canvas, brush handlers, paint texture) — ML mask supersedes manual painting.
- Update `README.md` to remove the "no ML models" claim.

### Out of scope
- Changing render resolution (stays 960×540).
- Changing the FX layer surface (jelly / water / cloth / refraction sliders are preserved).
- Server-side processing or model hosting (model files served from MediaPipe CDN initially; can be vendored later).
- Multi-person fine control (mask is single channel; multiclass output is collapsed to person/background).
- Recording / export.

## 3. User-visible changes

### Removed
- Compositor mode toggle "Full body / Paint" — only one mode now.
- Paint Mask panel section (brush, eraser, brush size, preview slider, clear button).
- Pointer drawing on the stage canvas.

### Added
- Mask refinement controls in the Compositor section:
  - **Edge feather** (0–1): controls the spatial smoothing radius of the mask.
  - **Mask stability** (0–1): EMA blend factor between the previous frame's refined mask and the current one. Higher = more temporally stable, lower = more responsive.
  - **Inpaint fallback** (0–1): blend weight between temporal plate and push-pull inpainted background where plate confidence is low.
- New debug view option: **Inpaint** — visualizes the push-pull output alone.

### Preserved
- Demo / Camera / Stop buttons.
- Reset plate button (now meaningfully useful: with ML mask, plate quality is bounded by what the camera has actually seen of the background; pressing reset before the user enters frame guarantees a clean plate).
- Invisibility strength slider.
- Follow lock slider — repurposed as the dilation radius applied to the ML mask before refinement (covers segmentation under-coverage near hair / loose clothing).
- Plate learning slider.
- Jelly / Water / Cloth / Refraction sliders.
- Edge gain slider.
- Diagnostics: Final / Matte / Plate / Inpaint debug views.

## 4. Architecture

### 4.1 Per-frame data flow

```
                    camera frame (960×540 RGBA)
                              │
        ┌─────────────────────┼──────────────────────────────────┐
        │                     │                                  │
        ↓                     ↓                                  │
  Pass 0: Segmenter     Pass A: Mask Refine ←──── (prev refined mask)
  MediaPipe Tasks       upsample + guided filter +
  ImageSegmenter        morphological close +
  → mask 256×256 R8     dilation (followLock) +
                        temporal EMA (maskStability)
                              │
                              ↓
                       refined mask (960×540 R8)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ↓                     ↓                     │
  Pass B: Plate Learn   Pass C: Push-Pull           │
  weight = (1 - mask)   inpainting (6 levels        │
  × plateLearning       down + up)                  │
  ping-pong plate       on plate × confidence       │
  + plate confidence    → inpainted bg (960×540)    │
                              │                     │
        ┌─────────────────────┴─────────────────────┘
        │                     │              │
        ↓                     ↓              ↓
                    Pass D: Composite
        select source = mix(plate, pushPull, inpaintFallback × (1 − plateConfidence))
        + jelly/water/cloth displacement (mask-gated)
        + edge glow
        final = mix(frame, reconstructed, mask × opacity)
                              │
                              ↓
                         canvas (BGRA8)
```

### 4.2 Pass-by-pass contract

#### Pass 0 — Segmenter (TypeScript, `src/lib/segmenter.ts`)
- **Input**: `HTMLCanvasElement` containing the current 960×540 frame.
- **Output**: `segment(source, timestamp): OffscreenCanvas | null` — runs `segmentForVideo` and writes person confidence (0–1, encoded as 0–255) into a persistent 256×256 RGBA `OffscreenCanvas`. Returns the canvas for the caller to upload to GPU. Returns `null` if the segmenter has not finished `init()`.
- **Implementation**: `@mediapipe/tasks-vision` `ImageSegmenter` configured with `selfie-multiclass-256x256.tflite`, GPU delegate, `outputConfidenceMasks: true`, `outputCategoryMask: false`. Person confidence = `1 − background_class_confidence` (model returns per-class confidences; we collapse to person mask).
- **Latency**: 5–10 ms on M1/M2.
- **Failure mode**: model load failure → `getMaskCanvas()` returns `null`; renderer falls back to the previous frame's refined mask (and ultimately a no-op all-zero mask on first frame). No motion-based fallback — we deliberately remove that path.

#### Pass A — Mask Refine (`src/lib/shaders/maskRefine.wgsl`, compute, 8×8 workgroup)
- **Input bindings**:
  - `rawMaskTex`: 256×256 R8 — output of Pass 0, uploaded via `copyExternalImageToTexture`.
  - `frameTex`: 960×540 RGBA — current camera frame (used as guide image).
  - `prevRefinedMaskTex`: 960×540 R8 — refined mask from previous frame (ping-pong).
  - `refinedMaskOut`: 960×540 R8 storage — write target.
  - `params` uniform: `maskFeather`, `maskStability`, `followLock`.
- **Algorithm**:
  1. Bilinear upsample raw mask 256×256 → 960×540.
  2. Dilate by radius proportional to `followLock` (3×3 max sample, optionally a second pass for larger radii).
  3. Joint bilateral / guided filter using `frameTex` luminance as guide — this snaps mask edges onto image edges (hair, sleeves). Single-pass approximation: weight neighbor mask samples by `exp(-|Y_center − Y_neighbor|² / σ_r²)` over a small kernel. Kernel radius from `maskFeather`.
  4. Temporal EMA: `out = mix(current, prev, maskStability)` clamped 0–1.
- **Cost**: ~1 ms.

#### Pass B — Plate Learn (`src/lib/shaders/plateLearn.wgsl`, compute, 8×8 workgroup)
- **Input bindings**:
  - `frameTex`: current camera frame.
  - `oldPlateTex`: previous plate (ping-pong).
  - `refinedMaskTex`: output of Pass A.
  - `newPlateOut`: storage RGBA8 — next plate.
  - `params` uniform: `plateLearning`, `plateInitialized` flag.
- **Algorithm**:
  - On first call after `Reset plate`: `newPlate = frame` (or rely on the JS-side `copyTextureToTexture` initializer, same as today).
  - Otherwise: `weight = (1 − refinedMask) × plateLearning`; `newPlate.rgb = mix(oldPlate.rgb, frame.rgb, weight)`.
  - `newPlate.a` stores **plate confidence** — accumulated background observations: `newConfidence = saturate(oldConfidence + (1 − refinedMask) × confidenceGain − refinedMask × confidenceDecay)`. `confidenceGain` and `confidenceDecay` are constants (e.g. 0.02 and 0.0 respectively — confidence does not decay just because someone is in front, only `Reset plate` clears it). This 4-channel plate replaces the current 3-channel one.
- **Cost**: <1 ms.

#### Pass C — Push-Pull Pyramid (`src/lib/shaders/pushPull.wgsl`, compute)

Two entry points (`@compute fn pushDown` and `@compute fn pushUp`) sharing the shader file. Six pyramid levels (960×540, 480×270, 240×135, 120×68, 60×34, 30×17). Each level is an RGBA8 storage texture (RGB = color, A = confidence weight).

- **Down pass (level i → level i+1)**: each output pixel = confidence-weighted average of the 4 corresponding input pixels:
  ```
  acc = sum(rgb × weight)
  w   = sum(weight)
  out.rgb = acc / max(w, eps)
  out.a   = clamp(w / 4.0, 0, 1)
  ```
  Initial level (level 0) = `plate.rgb × plate.a`, with `plate.a` as weight. Mask-occluded pixels (`refinedMask > threshold` AND `plate.a < threshold`) start with weight 0.
- **Up pass (level i+1 → level i)**: for each pixel, if its own confidence ≥ 0.95, keep it; else blend with bilinearly upsampled coarser level:
  ```
  coarse = bilinear_sample(level_i+1)
  out.rgb = mix(coarse.rgb, current.rgb, smoothstep(0.0, 0.95, current.a))
  out.a   = max(current.a, coarse.a × 0.9)
  ```
- **Output**: `inpaintTex` = level 0 after up pass — same dims as plate, every pixel has a defined RGB even if it was originally a hole.
- **Cost**: ~2 ms total across 12 dispatches.

Storage textures: a fixed-size array of 6 RGBA8 textures created once at init.

#### Pass D — Composite (`src/lib/shaders/composite.wgsl`, vertex + fragment, full-screen triangle)
- **Input bindings**:
  - `linearSampler`.
  - `frameTex`, `plateTex` (4-channel, alpha = confidence), `inpaintTex`, `refinedMaskTex`.
  - `params` uniform: all FX sliders + `debugView` enum + `inpaintFallback`.
- **Algorithm**:
  ```
  mask = sample(refinedMaskTex, uv).r

  // FX displacement (existing jelly/water/cloth math, unchanged)
  displacement = (jellyOffset + waterOffset + clothOffset) * refraction * mask

  // Sample background candidates at displaced uv
  plateSamp   = sample(plateTex,   uv + displacement)        // .rgb = color, .a = confidence
  inpaintSamp = sample(inpaintTex, uv + displacement)        // .rgb = color (alpha unused at level 0)

  // Prefer plate where confident, fall back to push-pull inpaint where not
  fillBlend = inpaintFallback * (1 - plateSamp.a)
  bg = mix(plateSamp.rgb, inpaintSamp.rgb, fillBlend)

  // Material light layer (existing caustic/weave math)
  materialLight = mask * (caustic*0.1 + weave*0.055 + jelly*0.045)

  reconstructed = bg + materialLight
  final = mix(sample(frameTex, uv).rgb, reconstructed, mask * opacity)

  // Edge glow (computed on the unblended mask, no displacement)
  edge = smoothstep(0.12, 0.48, mask) * (1 - smoothstep(0.52, 0.9, mask))
  final += edgeGlowColor * edge * edgeGain

  return vec4f(final, 1.0)
  ```
- Debug view branches as today (final / matte / plate / inpaint).
- **Cost**: <1 ms.

### 4.3 Module structure

```
src/lib/
  segmenter.ts          (NEW) MediaPipe wrapper
  webgpuRenderer.ts     (REWRITTEN) multi-pass orchestrator, no inline shaders
  shaders/              (NEW dir)
    maskRefine.wgsl
    plateLearn.wgsl
    pushPull.wgsl
    composite.wgsl
  types.ts              (CHANGED) FX settings shape
  camera.ts             (UNCHANGED)
  demoSource.ts         (UNCHANGED)
src/
  App.tsx               (CHANGED) paint UI removed, new sliders added
  App.css               (CHANGED) drop paint-canvas styles
README.md               (CHANGED) remove "no ML models" sentence; document new pipeline
```

WGSL files imported via Vite's `?raw` query (`import code from './shaders/maskRefine.wgsl?raw'`) so that shader code lives in `.wgsl` files (better tooling) but ships as inline strings without a separate fetch.

### 4.4 Texture & buffer inventory

| Resource | Format | Size | Lifetime |
|---|---|---|---|
| `videoTexture` | RGBA8 | 960×540 | persistent |
| `rawMaskTexture` | R8 | 256×256 | persistent |
| `refinedMask[2]` (ping-pong) | R8 | 960×540 | persistent |
| `plate[2]` (ping-pong, RGB+confidence) | RGBA8 | 960×540 | persistent |
| `pyramid[6]` | RGBA8 | 960×540 → 30×17 | persistent |
| `inpaintTexture` | alias of `pyramid[0]` after up-pass | RGBA8 | persistent |
| `uniformBuffer` | uniform | 64 B (16 floats) | persistent |

Total VRAM ≈ 6 MB. Trivial.

### 4.5 Execution order per frame

```
1. drawImage(video, frameCanvas)           // existing
2. segmenter.segment(frameCanvas)          // ~5–10 ms, returns when done (await)
3. encoder = device.createCommandEncoder()
4. queue.copyExternalImageToTexture(frameCanvas → videoTexture)
5. queue.copyExternalImageToTexture(maskCanvas  → rawMaskTexture)
6. queue.writeBuffer(uniformBuffer, settings)
7. encoder: Pass A (mask refine, ping-pong)
8. encoder: Pass B (plate learn, ping-pong)
9. encoder: Pass C — 6× pushDown + 6× pushUp
10. encoder: Pass D (composite to swapchain)
11. queue.submit
```

The `await segmenter.segment(...)` adds latency but the alternative (running segmenter on a previous frame) introduces a 1-frame mask delay that's visible on fast head turns. Start with the awaited version; if frame budget is tight we can switch to N-1 mask later as a setting.

## 5. State changes

### 5.1 `FxSettings`

Removed: `brushSize`, `brushMode`, `paintPreview`, and the `'paint'` variant of `mode`. The `mode` field is removed entirely.

Added: `maskFeather: number` (0–1, default 0.4), `maskStability: number` (0–1, default 0.6), `inpaintFallback: number` (0–1, default 0.85).

Repurposed: `followLock` becomes the ML-mask dilation radius (still 0–1, still default 0.82 — wider coverage for hair/loose clothing).

### 5.2 `DebugView`

Removed: `'paint'`. Added: `'inpaint'`. Final list: `'final' | 'matte' | 'plate' | 'inpaint'`.

### 5.3 Renderer state

`WebGpuFxRenderer` gains:
- a held reference to the `Segmenter` instance (or accepts the segmenter externally and only consumes its output texture — final choice in the implementation plan; leaning external for testability).
- `pyramidTextures: GPUTexture[]` (length 6).
- `refinedMaskTextures: GPUTexture[]` (length 2, ping-pong).
- `plateTextures` becomes RGBA8 (was already RGBA8) but the alpha channel now carries plate confidence rather than being unused.
- `frameNumber`, `plateInitialized`, ping-pong indices stay.

Loses: `paintTexture`, `emptyPaint`, `uploadPaint`, `uploadEmptyPaint`.

## 6. Error handling & edge cases

- **MediaPipe model fetch fails / WASM init fails**: surface in the existing `RendererStatus` strip. Pipeline still runs with an all-zero mask (effect becomes a no-op pass-through) — better than crashing the canvas.
- **Tab loses focus → segmenter `segmentForVideo` may timestamp-skip**: documented behavior; we pass `performance.now()` as the timestamp, MediaPipe handles drops.
- **First frame after `Reset plate`**: `plateInitialized = false` triggers `copyTextureToTexture(video → plate[0])`, plate confidence cleared to 0; push-pull will carry the brunt for the first few frames until plate accumulates.
- **Camera resolution ≠ 960×540**: existing pipeline already letterboxes via `drawImage`; unchanged.
- **WebGPU not supported**: existing error path unchanged.
- **User in shot from t=0 with no clean background ever**: plate confidence stays low everywhere the user has been, push-pull does the heavy lifting. Quality degrades gracefully — body silhouette still cut out, but background looks blurred/smeared in occluded regions. This is the best achievable without a clean plate; documented in README.

## 7. Performance budget (M1/M2 MacBook, 960×540)

| Stage | Target | Notes |
|---|---|---|
| MediaPipe segment | 5–10 ms | dominates |
| `copyExternalImageToTexture` ×2 | <0.5 ms | |
| Pass A (mask refine) | ~1 ms | |
| Pass B (plate learn) | <0.5 ms | |
| Pass C (push-pull, 12 dispatches) | ~2 ms | |
| Pass D (composite) | <1 ms | |
| **Total** | **~10–15 ms** | comfortable 60 fps |

If budget is exceeded on lower-end hardware, the first lever is running segmentation every other frame (mask interpolation in Pass A handles the gap). Not building this in v1 — measure first.

## 8. Testing

This project has no test infrastructure today. We are not adding a test framework as part of this change. Verification is:

- `npm run lint` clean.
- `npm run build` clean (TypeScript + Vite production build).
- Manual verification in a WebGPU-capable Chromium browser:
  - Demo source: person silhouette in the demo loop is masked and replaced; FX sliders behave.
  - Live camera: walking into frame works without prior background capture; standing still does not cause re-emergence (this is the key regression check vs. the old algorithm); fast head turns do not leave persistent ghosts.
  - All four debug views render something sensible.
  - Resetting plate while in frame → plate confidence drops, push-pull fallback kicks in immediately, no flash of original frame.
  - All sliders respond at 60 fps without UI hitching.

## 9. Open questions deferred to implementation

- Exact MediaPipe model variant (`selfie-multiclass-256x256` vs `selfie-segmenter`). Default to multiclass for the option of future per-region effects; switch if it costs >2 ms more than the segmenter variant.
- WGSL `?raw` import vs inline string. Default to `?raw` for editor tooling; fall back to inline if Vite/TS friction appears.
- Whether segmentation runs awaited (current frame) or one frame behind (parallel with previous frame's GPU work). Default awaited; switch behind a setting only if measured latency demands.

## 10. Out of scope, deferred

- Persisting plate across page reloads.
- Recording / export of the composited video.
- Multi-person mask channels.
- Mobile / touch UI.
- WebGL fallback for browsers without WebGPU.
- Vendoring the MediaPipe model files locally (currently fetched from MediaPipe CDN).
