struct Params {
  resolution: vec2f,
  time: f32,
  opacity: f32,
  jelly: f32,
  water: f32,
  cloth: f32,
  refraction: f32,
  edgeGain: f32,
  debugView: f32,
  inpaintFallback: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var frameTex: texture_2d<f32>;
@group(0) @binding(2) var plateTex: texture_2d<f32>;
@group(0) @binding(3) var inpaintTex: texture_2d<f32>;
@group(0) @binding(4) var refinedMaskTex: texture_2d<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@vertex
fn vertexMain(@builtin(vertex_index) index: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );
  let pos = positions[index];
  var out: VertexOut;
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = vec2f((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
  return out;
}

fn sampleSafe(tex: texture_2d<f32>, uv: vec2f) -> vec4f {
  return textureSample(tex, linearSampler, clamp(uv, vec2f(0.001), vec2f(0.999)));
}

@fragment
fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  let uv = input.uv;
  let frame = sampleSafe(frameTex, uv);
  let mask = sampleSafe(refinedMaskTex, uv).r;

  if (params.debugView > 0.5 && params.debugView < 1.5) {
    return vec4f(vec3f(mask), 1.0);
  }
  if (params.debugView > 1.5 && params.debugView < 2.5) {
    return vec4f(sampleSafe(plateTex, uv).rgb, 1.0);
  }
  if (params.debugView > 2.5) {
    return vec4f(sampleSafe(inpaintTex, uv).rgb, 1.0);
  }

  let t = params.time;
  let aspect = params.resolution.x / params.resolution.y;
  let centered = vec2f((uv.x - 0.5) * aspect, uv.y - 0.5);
  let jellyOffset = vec2f(
    sin((uv.y * 24.0) + t * 4.0) * 0.012,
    sin((uv.x * 18.0) - t * 2.7) * 0.007
  ) * params.jelly;
  let waterOffset = vec2f(
    sin((centered.x * 40.0) + t * 5.2) + sin((uv.y * 65.0) - t * 3.4),
    cos((centered.y * 38.0) - t * 4.6)
  ) * 0.006 * params.water;
  let clothOffset = vec2f(
    sin((uv.y * 92.0) + t * 1.6) * 0.003,
    cos((uv.x * 76.0) - t * 1.2) * 0.003
  ) * params.cloth;
  let displacement = (jellyOffset + waterOffset + clothOffset) * params.refraction * mask;
  let plate = sampleSafe(plateTex, uv + displacement);
  let inpaint = sampleSafe(inpaintTex, uv + displacement);
  let bodyFill = smoothstep(0.12, 0.72, mask);
  let fillBlend = clamp(max(params.inpaintFallback * bodyFill, params.inpaintFallback * (1.0 - plate.a)), 0.0, 1.0);
  let bg = mix(plate.rgb, inpaint.rgb, fillBlend);

  let caustic = (sin((uv.x + uv.y) * 80.0 + t * 5.0) * 0.5 + 0.5) * params.water;
  let weave = (sin(uv.x * 180.0) * sin(uv.y * 150.0) * 0.5 + 0.5) * params.cloth;
  let materialLight = bodyFill * (caustic * 0.06 + weave * 0.035 + params.jelly * 0.025);
  var finalColor = mix(frame.rgb, bg + materialLight, bodyFill * params.opacity);

  let edge = smoothstep(0.12, 0.48, mask) * (1.0 - smoothstep(0.52, 0.9, mask));
  finalColor = finalColor + vec3f(0.52, 0.94, 1.0) * edge * params.edgeGain;
  return vec4f(finalColor, 1.0);
}
