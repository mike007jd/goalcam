# InvisibleCam

InvisibleCam is a small experimental webcam FX project and a practical testbed for Codex / Goal-driven development. The maintainer built it to see how far an agentic coding workflow could carry a real-time WebGPU idea, and is very happy with the result.

Open-source browser webcam FX engine for MacBook-class hardware. It uses a pure WebGPU/WGSL shader pipeline for:

- real-time full-body invisibility through shader-side horizontal-dominant local fill
- MediaPipe person segmentation for stable body masks
- selectable MediaPipe selfie classes: hair, body skin, face skin, clothes, and accessories
- left and right neighboring pixels diffused into each masked pixel, with low-weight surrounding softening and push-pull fallback
- mutually exclusive Jelly, Water, and Cloth modes driven by the selected person cutout
- mask dilation, feathering, temporal stability, and adjustable invisibility strength

## Run

```bash
npm install
npm run dev
```

Open the Vite localhost URL in a WebGPU-capable Chromium browser. Camera capture requires localhost or HTTPS.

## Shader Pipeline

InvisibleCam loads MediaPipe Tasks Vision in the browser and runs the rest of the effect in WebGPU. Each frame flows through person-class segmentation, mask refinement, horizontal-dominant local fill, push-pull fallback inpainting, and final compositing.

This is not temporal background reconstruction. The person mask is treated as a hole; each masked pixel searches its own nearby left and right source pixels, then uses surrounding samples only as a softening layer. Body-material FX are mask-local modes: the selected person cutout provides the visible body volume and lighting, while the hidden fill is only a background guard. Use the diagnostics selector to inspect Final, Matte, and Inpaint views.

## Third-Party Segmentation

This project does not ship a custom segmentation model. Person masks come from Google's MediaPipe Tasks Vision package and the hosted Selfie Multiclass image segmenter model. InvisibleCam contributes the browser app, mask controls, WebGPU fill shaders, compositing pipeline, QA harness, and UX around that model.

MediaPipe Tasks Vision is an Apache-2.0 dependency. The app downloads MediaPipe WASM and the segmentation model from public Google/CDN URLs at runtime. Keep this distinction clear in issues, demos, and contributions: segmentation is third-party; InvisibleCam's own work starts after the mask is produced.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) before opening a change. In short: keep the app browser-only, preserve third-party attribution, document any model or asset source, and run the local verification commands before sending a patch.

## Verification

```bash
npm run lint
npm run build
```

For local static-image checks, open:

```text
http://localhost:5173/qa-image-test.html
```
