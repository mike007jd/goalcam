struct Params {
  resolution: vec2f,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var frameTex: texture_2d<f32>;
@group(0) @binding(1) var refinedMaskTex: texture_2d<f32>;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> params: Params;

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
  let weight = 1.0 - hole;
  textureStore(outTex, pixel, vec4f(frame, weight));
}
