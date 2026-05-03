# Push-Pull Invisibility Implementation Plan

## Scope

Implement the corrected pure spatial fill direction for InvisibleCam:

`inpaintFill = localDirectionalFill(frame, hole = personMask, fallback = pushPull)`

`final = mix(frame, inpaintFill, featheredMask)`

## Implementation Tasks

1. Rewrite `src/lib/shaders/pyramidBase.wgsl` so level 0 reads the original frame and refined mask. Person-mask pixels become holes by writing zero fill weight.
2. Simplify `src/lib/shaders/composite.wgsl` so final rendering only mixes frame, inpaint fill, and refined mask. Optional distortion stays mask-gated and defaults to zero.
3. Remove the old background-learning resources and dispatches from `src/lib/webgpuRenderer.ts`.
4. Reduce renderer bind groups from background-state plus mask combinations to mask-only combinations.
5. Wait for both WebGPU and MediaPipe initialization before starting the render loop.
6. Remove obsolete controls from the UI: reset, learning rate, fallback blend, and fabric distortion.
7. Keep the controls that map to the corrected product behavior: invisibility strength, mask coverage, edge feather, mask stability, selectable person-class mask target, mutually exclusive body FX mode, edge, and debug view.
8. Update README and local docs so future work follows the Snapchat-style local directional fill direction.

## Verification

Run:

```bash
npm run lint
npm run build
```

Then start Vite and verify in browser:

1. Test mode fills the silhouette with surrounding colors.
2. Camera mode takes effect immediately with no need to leave the frame first.
3. Standing still remains effective.
4. No regular horizontal striping appears.
5. Matte shows a white person mask over black background.
6. Inpaint shows the directional fill result with push-pull fallback.
7. Invisibility strength 0 shows the original frame; 1 shows full fill inside the mask.
8. Mask target toggles can isolate hair, skin, face, clothes, and accessories.
9. Body FX modes only affect pixels inside the selected person classes.
