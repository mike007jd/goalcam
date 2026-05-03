# InvisibleCam Push-Pull Invisibility Design

## Goal

InvisibleCam should create a Snapchat-style invisibility effect, not a true clean-background reconstruction. The person mask is treated as a hole; each masked pixel samples nearby unmasked pixels around its own position, then weaker areas fall back to surrounding-pixel blur so the subject visually melts into nearby wall, floor, jacket, and lighting colors.

## Core Algorithm

Each frame uses this flow:

1. Draw the current camera or demo frame.
2. Run MediaPipe person segmentation.
3. Build a selected person-class mask from MediaPipe categories: hair, body skin, face skin, clothes, and optional accessories.
4. Refine the mask with coverage, feathering, and temporal stability controls.
5. Build a push-pull pyramid from the original frame, with mask pixels set to zero fill weight.
6. Build a local directional fill around each masked pixel, using push-pull as fallback.
7. Composite the original frame with the inpaint fill using the feathered mask.

The final visual equation is:

`final = mix(frame, localDirectionalFill(frame, hole = personMask), featheredMask * opacity)`

## Product Behavior

- The effect must work immediately when the user starts in frame.
- Standing still should not make the person reappear.
- Debug views are Final, Matte, and Inpaint.
- Jelly, Water, and Cloth are mutually exclusive body-material modes. They are applied only inside the person mask, using the person cutout as the visible body volume and the invisible fill only as a background guard; edge highlight remains optional and defaults to zero.
- Mask target is split into hair, body skin, face skin, clothes, and accessories. Default excludes accessories and keeps mask coverage moderate to avoid swallowing adjacent chairs or props.

## Non-Goals

- Recording and export.
- Changing camera or demo source capture.
- Reconstructing a clean hidden background.
