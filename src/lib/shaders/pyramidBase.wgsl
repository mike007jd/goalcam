struct Params {
  resolution: vec2f,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var frameTex: texture_2d<f32>;
@group(0) @binding(1) var plateTex: texture_2d<f32>;
@group(0) @binding(2) var refinedMaskTex: texture_2d<f32>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= u32(params.resolution.x) || xy.y >= u32(params.resolution.y)) {
    return;
  }

  let pixel = vec2i(xy);
  let frame = textureLoad(frameTex, pixel, 0);
  let plate = textureLoad(plateTex, pixel, 0);
  let mask = textureLoad(refinedMaskTex, pixel, 0).r;
  let softMask = smoothstep(0.18, 0.72, mask);
  let outsideWeight = 1.0 - softMask;
  let plateWeight = plate.a * outsideWeight * 0.25;
  let frameWeight = outsideWeight;
  let rgb = (frame.rgb * frameWeight + plate.rgb * plateWeight) / max(frameWeight + plateWeight, 0.001);
  textureStore(outTex, pixel, vec4f(rgb, outsideWeight));
}
