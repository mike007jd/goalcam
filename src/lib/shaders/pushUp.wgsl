struct DstSize {
  size: vec2f,
};

@group(0) @binding(0) var currentTex: texture_2d<f32>;
@group(0) @binding(1) var coarseTex: texture_2d<f32>;
@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var linearSampler: sampler;
@group(0) @binding(4) var<uniform> dst: DstSize;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= u32(dst.size.x) || xy.y >= u32(dst.size.y)) {
    return;
  }

  let pixel = vec2i(xy);
  let uv = (vec2f(xy) + 0.5) / dst.size;
  let current = textureLoad(currentTex, pixel, 0);
  let coarse = textureSampleLevel(coarseTex, linearSampler, uv, 0.0);
  let keep = smoothstep(0.0, 0.95, current.a);
  let rgb = mix(coarse.rgb, current.rgb, keep);
  let conf = max(current.a, coarse.a * 0.9);
  textureStore(outTex, pixel, vec4f(rgb, conf));
}
