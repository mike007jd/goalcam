# Contributing to InvisibleCam

InvisibleCam is an experimental browser webcam FX engine and a test project for Codex / Goal-driven development. The maintainer is very happy with how this workflow has moved the project forward, and contributions should keep that spirit: small, clear, evidence-backed changes that improve the actual live camera experience.

## Project Boundary

InvisibleCam is not a custom ML segmentation project. It uses Google's MediaPipe Tasks Vision package plus the hosted Selfie Multiclass image segmenter model to produce person-class masks. The project code then handles:

- selectable mask classes: hair, body skin, face skin, clothes, accessories
- mask refinement, feathering, temporal stability, and coverage controls
- WebGPU/WGSL push-pull and local fill shaders
- mask-local body FX modes
- browser UI, camera/demo modes, docs, and QA harnesses

Do not describe the segmentation model as authored by this repository.

## Third-Party Attribution

Current third-party segmentation pieces:

- `@mediapipe/tasks-vision`, licensed Apache-2.0
- MediaPipe WASM loaded from `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm`
- Selfie Multiclass model loaded from Google's public model storage

If you change any model, runtime, CDN, asset, or QA image, document:

- source URL or package name
- license
- labels/categories used by the app
- privacy or runtime implications
- why the replacement is better for this project

QA images must have permissive usage rights and be listed in `public/qa/README.md`.

## Good Contribution Areas

- Mask quality and class selection behavior
- Real-time WebGPU fill and compositing quality
- Body-material FX that work only inside the selected mask
- Camera startup, browser compatibility, and performance
- Clear UI controls for product behavior
- QA samples and reproducible visual checks
- Documentation that explains what the project actually does

## Product Rules

- Default target is person-only, not generic subject segmentation.
- Do not let adjacent chairs or props become invisible by default.
- Body FX should be mask-local. They must not affect the background.
- Invisibility fill is spatial and live; do not reintroduce temporal background plate learning unless the product direction changes.
- Keep the app browser-only. No server-side frame upload is expected.

## Development

Install and run:

```bash
npm install
npm run dev
```

Required checks:

```bash
npm run lint
npm run build
```

For visual QA:

```text
http://localhost:5173/qa-image-test.html
```

For shader or segmentation changes, verify at least:

- Final view
- Matte view
- Inpaint view
- Person core mask
- Clothes-only or skin-only mask
- Each body FX mode touched by the change

## Pull Request Expectations

Keep PRs focused. Explain:

- what changed
- why it changed
- how it affects the live camera experience
- what was verified
- any third-party dependency or asset involved

Do not bundle unrelated UI redesigns, model swaps, and shader rewrites in one PR unless the change is intentionally scoped as a larger product pass.
