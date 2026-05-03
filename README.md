# InvisibleCam

Open-source browser webcam FX engine for MacBook-class hardware. It uses a pure WebGPU/WGSL shader pipeline for:

- real-time full-body invisibility through shader-side horizontal-dominant local fill
- MediaPipe person segmentation for stable body masks
- left and right neighboring pixels diffused into each masked pixel, with low-weight surrounding softening and push-pull fallback
- optional jelly and water distortion layers over the inpaint fill
- mask dilation, feathering, temporal stability, and adjustable invisibility strength

## Run

```bash
npm install
npm run dev
```

Open the Vite localhost URL in a WebGPU-capable Chromium browser. Camera capture requires localhost or HTTPS.

## Shader Pipeline

InvisibleCam loads MediaPipe Tasks Vision in the browser and runs the rest of the effect in WebGPU. Each frame flows through person segmentation, mask refinement, horizontal-dominant local fill, push-pull fallback inpainting, and final compositing.

This is not temporal background reconstruction. The person mask is treated as a hole; each masked pixel searches its own nearby left and right source pixels, then uses surrounding samples only as a softening layer. Push-pull blur fills areas where local samples are weak. Use the diagnostics selector to inspect Final, Matte, and Inpaint views.

## Verification

```bash
npm run lint
npm run build
```

For local static-image checks, open:

```text
http://localhost:5173/qa-image-test.html
```
