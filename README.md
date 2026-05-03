# InvisibleCam

Open-source browser webcam FX engine for MacBook-class hardware. It uses a pure WebGPU/WGSL shader pipeline for:

- real-time full-body invisibility against a temporally reconstructed background plate
- MediaPipe person segmentation for stable body masks
- push-pull shader inpainting when the background plate has low confidence
- stackable jelly, water, and cloth distortion layers over the reconstructed plate
- mask dilation, feathering, temporal stability, and adjustable invisibility strength

## Run

```bash
npm install
npm run dev
```

Open the Vite localhost URL in a WebGPU-capable Chromium browser. Camera capture requires localhost or HTTPS.

## Shader Pipeline

InvisibleCam loads MediaPipe Tasks Vision in the browser and runs the rest of the effect in WebGPU. Each frame flows through person segmentation, mask refinement, temporal plate learning, push-pull inpainting, and final compositing with the existing jelly, water, cloth, refraction, and edge layers.

Use the diagnostics selector to inspect Final, Matte, Plate, and Inpaint views. Reset plate is most useful before entering frame; if the user starts in shot, the inpaint fallback keeps the body cutout usable while the plate confidence improves.

## Verification

```bash
npm run lint
npm run build
```
