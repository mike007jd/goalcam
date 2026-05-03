struct Params {
  resolution: vec2f,
  plateLearning: f32,
  pad0: f32,
};

@group(0) @binding(0) var frameTex: texture_2d<f32>;
@group(0) @binding(1) var oldPlateTex: texture_2d<f32>;
@group(0) @binding(2) var refinedMaskTex: texture_2d<f32>;
@group(0) @binding(3) var newPlateOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= u32(params.resolution.x) || xy.y >= u32(params.resolution.y)) {
    return;
  }

  let pixel = vec2i(xy);
  let frame = textureLoad(frameTex, pixel, 0);
  let oldPlate = textureLoad(oldPlateTex, pixel, 0);
  let mask = textureLoad(refinedMaskTex, pixel, 0).r;
  let learnGate = clamp(1.0 - mask, 0.0, 1.0);
  let learnRate = clamp(learnGate * params.plateLearning, 0.0, 1.0);
  let nextRgb = mix(oldPlate.rgb, frame.rgb, learnRate);
  let nextConf = clamp(oldPlate.a + learnGate * 0.04, 0.0, 1.0);

  textureStore(newPlateOut, pixel, vec4f(nextRgb, nextConf));
}
