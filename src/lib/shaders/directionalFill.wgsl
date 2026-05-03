struct Params {
  resolution: vec2f,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var frameTex: texture_2d<f32>;
@group(0) @binding(1) var refinedMaskTex: texture_2d<f32>;
@group(0) @binding(2) var pushPullTex: texture_2d<f32>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: Params;

const SOURCE_MASK_MAX: f32 = 0.055;
const MAX_HORIZONTAL_SEARCH: i32 = 360;
const HORIZONTAL_STEP: i32 = 5;

fn sampleSourceColor(pixel: vec2i) -> vec3f {
  let maxPixel = vec2i(i32(params.resolution.x) - 1, i32(params.resolution.y) - 1);
  var acc = vec3f(0.0);
  var total = 0.0;
  for (var dy = -6; dy <= 6; dy = dy + 1) {
    for (var dx = -4; dx <= 4; dx = dx + 1) {
      let p = clamp(pixel + vec2i(dx, dy), vec2i(0), maxPixel);
      let weight = 1.0 / (1.0 + f32(abs(dx)) * 0.34 + f32(abs(dy)) * 0.28);
      acc = acc + textureLoad(frameTex, p, 0).rgb * weight;
      total = total + weight;
    }
  }
  return acc / max(total, 0.001);
}

fn rowOffset(index: i32) -> i32 {
  if (index == 0) { return 0; }
  if (index == 1) { return -3; }
  if (index == 2) { return 3; }
  if (index == 3) { return -7; }
  if (index == 4) { return 7; }
  if (index == 5) { return -12; }
  if (index == 6) { return 12; }
  if (index == 7) { return -20; }
  return 20;
}

fn rowWeight(offset: i32) -> f32 {
  return 1.0 / (1.0 + f32(abs(offset)) * 0.1);
}

fn findHorizontalSource(pixel: vec2i, yOffset: i32, direction: i32) -> vec4f {
  let maxPixel = vec2i(i32(params.resolution.x) - 1, i32(params.resolution.y) - 1);
  let sy = pixel.y + yOffset;
  if (sy < 0 || sy > maxPixel.y) {
    return vec4f(0.0);
  }

  for (var offset = HORIZONTAL_STEP; offset <= MAX_HORIZONTAL_SEARCH; offset = offset + HORIZONTAL_STEP) {
    let sx = pixel.x + offset * direction;
    if (sx < 0 || sx > maxPixel.x) {
      break;
    }

    let samplePixel = vec2i(sx, sy);
    let sampleMask = textureLoad(refinedMaskTex, samplePixel, 0).r;
    if (sampleMask < SOURCE_MASK_MAX) {
      return vec4f(sampleSourceColor(samplePixel), f32(offset));
    }
  }
  return vec4f(0.0);
}

fn mergeLeftRight(left: vec4f, right: vec4f) -> vec4f {
  let hasLeft = left.a > 0.0;
  let hasRight = right.a > 0.0;
  if (hasLeft && hasRight) {
    let ratio = left.a / max(left.a + right.a, 1.0);
    let contrast = smoothstep(0.08, 0.42, length(left.rgb - right.rgb));
    let seamWidth = mix(0.43, 0.28, contrast);
    let t = smoothstep(0.5 - seamWidth, 0.5 + seamWidth, ratio);
    let distanceConfidence = 1.0 - smoothstep(220.0, 370.0, min(left.a, right.a));
    return vec4f(mix(left.rgb, right.rgb, t), mix(0.58, 0.86, distanceConfidence));
  }
  if (hasLeft) {
    let distanceConfidence = 1.0 - smoothstep(150.0, 360.0, left.a);
    return vec4f(left.rgb, mix(0.3, 0.56, distanceConfidence));
  }
  if (hasRight) {
    let distanceConfidence = 1.0 - smoothstep(150.0, 360.0, right.a);
    return vec4f(right.rgb, mix(0.3, 0.56, distanceConfidence));
  }
  return vec4f(0.0);
}

fn horizontalFill(pixel: vec2i) -> vec4f {
  var acc = vec3f(0.0);
  var total = 0.0;
  for (var row = 0; row < 9; row = row + 1) {
    let yOffset = rowOffset(row);
    let merged = mergeLeftRight(
      findHorizontalSource(pixel, yOffset, -1),
      findHorizontalSource(pixel, yOffset, 1)
    );
    let weight = merged.a * rowWeight(yOffset);
    acc = acc + merged.rgb * weight;
    total = total + weight;
  }

  if (total <= 0.001) {
    return vec4f(0.0);
  }

  let confidence = mix(0.46, 0.78, smoothstep(0.4, 3.6, total));
  return vec4f(acc / total, confidence);
}

fn ringOffset(index: i32, radius: f32) -> vec2i {
  if (index == 0) { return vec2i(i32(round(radius)), 0); }
  if (index == 1) { return vec2i(-i32(round(radius)), 0); }
  if (index == 2) { return vec2i(0, i32(round(radius))); }
  if (index == 3) { return vec2i(0, -i32(round(radius))); }
  if (index == 4) { return vec2i(i32(round(radius * 0.7071)), i32(round(radius * 0.7071))); }
  if (index == 5) { return vec2i(-i32(round(radius * 0.7071)), i32(round(radius * 0.7071))); }
  if (index == 6) { return vec2i(i32(round(radius * 0.7071)), -i32(round(radius * 0.7071))); }
  return vec2i(-i32(round(radius * 0.7071)), -i32(round(radius * 0.7071)));
}

fn softAroundFill(pixel: vec2i) -> vec4f {
  let maxPixel = vec2i(i32(params.resolution.x) - 1, i32(params.resolution.y) - 1);
  var acc = vec3f(0.0);
  var total = 0.0;
  for (var ring = 0; ring < 5; ring = ring + 1) {
    let radius = f32(18 + ring * 26);
    let distanceWeight = 1.0 / pow(1.0 + radius * 0.024, 2.0);
    for (var dir = 0; dir < 8; dir = dir + 1) {
      let samplePixel = pixel + ringOffset(dir, radius);
      if (samplePixel.x < 0 || samplePixel.y < 0 || samplePixel.x > maxPixel.x || samplePixel.y > maxPixel.y) {
        continue;
      }
      let sampleMask = textureLoad(refinedMaskTex, samplePixel, 0).r;
      let sourceWeight = 1.0 - smoothstep(0.025, 0.18, sampleMask);
      let w = sourceWeight * distanceWeight;
      acc = acc + sampleSourceColor(samplePixel) * w;
      total = total + w;
    }
  }

  if (total <= 0.001) {
    return vec4f(0.0);
  }
  return vec4f(acc / total, smoothstep(0.04, 0.45, total));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= u32(params.resolution.x) || xy.y >= u32(params.resolution.y)) {
    return;
  }

  let pixel = vec2i(xy);
  let frame = textureLoad(frameTex, pixel, 0).rgb;
  let mask = textureLoad(refinedMaskTex, pixel, 0).r;
  let hole = smoothstep(0.02, 0.28, mask);
  if (hole < 0.01) {
    textureStore(outTex, pixel, vec4f(frame, 0.0));
    return;
  }

  let pushPull = textureLoad(pushPullTex, pixel, 0).rgb;
  let horizontal = horizontalFill(pixel);
  let around = softAroundFill(pixel);
  if (horizontal.a <= 0.001 && around.a <= 0.001) {
    textureStore(outTex, pixel, vec4f(pushPull, 0.0));
    return;
  }

  var fill = mix(pushPull, horizontal.rgb, horizontal.a);
  fill = mix(fill, around.rgb, around.a * 0.1);

  let confidence = clamp(horizontal.a + around.a * 0.05, 0.42, 0.78);
  textureStore(outTex, pixel, vec4f(fill, confidence));
}
