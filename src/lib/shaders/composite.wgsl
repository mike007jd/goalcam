struct Params {
  resolution: vec2f,
  time: f32,
  opacity: f32,
  jelly: f32,
  water: f32,
  refraction: f32,
  edgeGain: f32,
  debugView: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
  pad4: f32,
};

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var frameTex: texture_2d<f32>;
@group(0) @binding(2) var inpaintTex: texture_2d<f32>;
@group(0) @binding(3) var directionalFillTex: texture_2d<f32>;
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

fn sampleDirectionalAt(uv: vec2f, offset: vec2f) -> vec4f {
  return textureSample(
    directionalFillTex,
    linearSampler,
    clamp(uv + offset / params.resolution, vec2f(0.001), vec2f(0.999))
  );
}

fn sampleDirectionalFill(uv: vec2f) -> vec4f {
  var acc = sampleDirectionalAt(uv, vec2f(0.0)) * 3.0;
  var total = 3.0;

  acc = acc + sampleDirectionalAt(uv, vec2f(5.0, 0.0)) * 2.0;
  acc = acc + sampleDirectionalAt(uv, vec2f(-5.0, 0.0)) * 2.0;
  acc = acc + sampleDirectionalAt(uv, vec2f(0.0, 5.0)) * 2.5;
  acc = acc + sampleDirectionalAt(uv, vec2f(0.0, -5.0)) * 2.5;
  total = total + 9.0;

  acc = acc + sampleDirectionalAt(uv, vec2f(0.0, 12.0)) * 1.5;
  acc = acc + sampleDirectionalAt(uv, vec2f(0.0, -12.0)) * 1.5;
  acc = acc + sampleDirectionalAt(uv, vec2f(12.0, 0.0));
  acc = acc + sampleDirectionalAt(uv, vec2f(-12.0, 0.0));
  total = total + 5.0;

  acc = acc + sampleDirectionalAt(uv, vec2f(8.0, 6.0)) * 0.75;
  acc = acc + sampleDirectionalAt(uv, vec2f(-8.0, 6.0)) * 0.75;
  acc = acc + sampleDirectionalAt(uv, vec2f(8.0, -6.0)) * 0.75;
  acc = acc + sampleDirectionalAt(uv, vec2f(-8.0, -6.0)) * 0.75;
  total = total + 3.0;

  return acc / total;
}

@fragment
fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  let uv = input.uv;
  let frame = textureSample(frameTex, linearSampler, uv).rgb;
  let mask = textureSample(refinedMaskTex, linearSampler, uv).r;
  let inpaint = textureSample(inpaintTex, linearSampler, uv).rgb;
  let directional = sampleDirectionalFill(uv);
  let fill = mix(inpaint, directional.rgb, directional.a);

  if (params.debugView > 0.5 && params.debugView < 1.5) {
    return vec4f(vec3f(mask), 1.0);
  }
  if (params.debugView > 1.5) {
    return vec4f(fill, 1.0);
  }

  let soft = smoothstep(0.02, 0.28, mask);
  let t = params.time;
  let jellyOffset = vec2f(
    sin((uv.y * 24.0) + t * 4.0) * 0.012,
    sin((uv.x * 18.0) - t * 2.7) * 0.007
  ) * params.jelly;
  let waterOffset = vec2f(
    sin((uv.x * 40.0) + t * 5.2),
    cos((uv.y * 38.0) - t * 4.6)
  ) * 0.006 * params.water;
  let displacement = (jellyOffset + waterOffset) * params.refraction * soft;
  let displacedFill = sampleDirectionalFill(clamp(uv + displacement, vec2f(0.001), vec2f(0.999)));
  let displacedFallback = textureSample(
    inpaintTex,
    linearSampler,
    clamp(uv + displacement, vec2f(0.001), vec2f(0.999))
  ).rgb;
  let finalFill = mix(displacedFallback, displacedFill.rgb, displacedFill.a);

  let blend = soft * params.opacity;
  var finalColor = mix(frame, finalFill, blend);

  let edge = soft * (1.0 - soft) * 4.0;
  finalColor = finalColor + vec3f(0.5, 0.85, 1.0) * edge * params.edgeGain;
  return vec4f(finalColor, 1.0);
}
