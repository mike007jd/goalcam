struct Params {
  resolution: vec2f,
  initialConfidence: f32,
  pad0: f32,
};

@group(0) @binding(0) var frameTex: texture_2d<f32>;
@group(0) @binding(1) var plateOut: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= u32(params.resolution.x) || xy.y >= u32(params.resolution.y)) {
    return;
  }

  let frame = textureLoad(frameTex, vec2i(xy), 0);
  textureStore(plateOut, vec2i(xy), vec4f(frame.rgb, params.initialConfidence));
}
