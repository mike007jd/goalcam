struct Params {
  resolution: vec2f,
  maskFeather: f32,
  maskStability: f32,
  followLock: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
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
  let res = vec2f(params.resolution);
  var mask = textureSampleLevel(rawMaskTex, linearSampler, uv, 0.0).r;

  let dilatePx = i32(round(params.followLock * 14.0));
  if (dilatePx > 0) {
    for (var dy = -dilatePx; dy <= dilatePx; dy = dy + 2) {
      for (var dx = -dilatePx; dx <= dilatePx; dx = dx + 2) {
        let sampleUv = clamp(uv + vec2f(f32(dx), f32(dy)) / res, vec2f(0.0), vec2f(1.0));
        mask = max(mask, textureSampleLevel(rawMaskTex, linearSampler, sampleUv, 0.0).r);
      }
    }
  }

  let radius = i32(round(params.maskFeather * 6.0));
  if (radius > 0) {
    let centerLuma = luma(textureLoad(frameTex, pixel, 0).rgb);
    let maxPixel = vec2i(i32(params.resolution.x) - 1, i32(params.resolution.y) - 1);
    var acc = mask;
    var total = 1.0;
    let sigmaR = 0.15;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
      for (var dx = -radius; dx <= radius; dx = dx + 1) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        let p = clamp(pixel + vec2i(dx, dy), vec2i(0), maxPixel);
        let nLuma = luma(textureLoad(frameTex, p, 0).rgb);
        let nUv = (vec2f(p) + 0.5) / res;
        let nMask = textureSampleLevel(rawMaskTex, linearSampler, nUv, 0.0).r;
        let dl = nLuma - centerLuma;
        let weight = exp(-(dl * dl) / (sigmaR * sigmaR));
        acc = acc + nMask * weight;
        total = total + weight;
      }
    }
    mask = acc / total;
  }

  let prev = textureLoad(prevRefinedTex, pixel, 0).r;
  let finalMask = mix(mask, prev, clamp(params.maskStability, 0.0, 0.95));
  textureStore(refinedOut, pixel, vec4f(clamp(finalMask, 0.0, 1.0), 0.0, 0.0, 1.0));
}
