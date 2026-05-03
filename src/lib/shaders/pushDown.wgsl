struct DstSize {
  size: vec2f,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var dstTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> dst: DstSize;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let xy = vec2u(id.xy);
  if (xy.x >= u32(dst.size.x) || xy.y >= u32(dst.size.y)) {
    return;
  }

  let base = vec2i(xy * 2u);
  let srcSize = vec2i(textureDimensions(srcTex));
  var acc = vec3f(0.0);
  var weight = 0.0;
  for (var y = 0; y < 2; y = y + 1) {
    for (var x = 0; x < 2; x = x + 1) {
      let s = textureLoad(srcTex, clamp(base + vec2i(x, y), vec2i(0), srcSize - vec2i(1)), 0);
      acc = acc + s.rgb * s.a;
      weight = weight + s.a;
    }
  }

  let rgb = acc / max(weight, 0.001);
  textureStore(dstTex, vec2i(xy), vec4f(rgb, clamp(weight * 0.25, 0.0, 1.0)));
}
