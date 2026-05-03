struct Params {
  resolution: vec2f,
  time: f32,
  opacity: f32,
  bodyFxMode: f32,
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

fn sampleHiddenFill(uv: vec2f) -> vec3f {
  let inpaint = textureSample(inpaintTex, linearSampler, uv).rgb;
  let directional = sampleDirectionalFill(uv);
  return mix(inpaint, directional.rgb, directional.a);
}

fn sampleMaskAt(uv: vec2f, offset: vec2f) -> f32 {
  return textureSample(
    refinedMaskTex,
    linearSampler,
    clamp(uv + offset / params.resolution, vec2f(0.001), vec2f(0.999))
  ).r;
}

fn maskNormal(uv: vec2f) -> vec3f {
  let dx = sampleMaskAt(uv, vec2f(2.0, 0.0)) - sampleMaskAt(uv, vec2f(-2.0, 0.0));
  let dy = sampleMaskAt(uv, vec2f(0.0, 2.0)) - sampleMaskAt(uv, vec2f(0.0, -2.0));
  return normalize(vec3f(-dx * 2.8, -dy * 2.8, 1.0));
}

fn jellyMaterial(uv: vec2f, soft: f32) -> vec3f {
  let t = params.time;
  let normal = maskNormal(uv);
  let wobble = vec2f(
    sin((uv.y * 28.0) + t * 4.1) + sin((uv.x + uv.y) * 18.0 - t * 2.3),
    cos((uv.x * 22.0) - t * 3.4)
  ) * 7.5;
  let refractUv = clamp(uv + ((normal.xy * 9.0 + wobble) / params.resolution) * soft, vec2f(0.001), vec2f(0.999));
  let refracted = sampleHiddenFill(refractUv);
  let edge = soft * (1.0 - soft) * 4.0;
  let pulse = sin(t * 3.0 + uv.y * 20.0) * 0.5 + 0.5;
  let volume = smoothstep(0.18, 0.9, soft);
  let body = mix(refracted, refracted * vec3f(0.78, 1.05, 0.96) + vec3f(0.12, 0.48, 0.42), 0.24 * volume);
  return body + vec3f(0.28, 1.0, 0.82) * (edge * 0.2 + pulse * volume * 0.035);
}

fn waterMaterial(uv: vec2f, soft: f32) -> vec3f {
  let t = params.time;
  let waveA = sin((uv.x * 46.0) + (uv.y * 10.0) + t * 4.8);
  let waveB = cos((uv.y * 42.0) - (uv.x * 8.0) - t * 3.7);
  let ripple = sin((uv.x + uv.y) * 76.0 + t * 6.2);
  let normal = maskNormal(uv);
  let flow = vec2f(waveA + ripple * 0.45, waveB - ripple * 0.35) * 9.0 + normal.xy * 12.0;
  let refracted = sampleHiddenFill(clamp(uv + (flow / params.resolution) * soft, vec2f(0.001), vec2f(0.999)));
  let caustic = pow(sin((uv.x * 90.0) + waveB * 1.4 + t * 2.1) * 0.5 + 0.5, 5.0);
  let edge = soft * (1.0 - soft) * 4.0;
  let liquid = refracted * vec3f(0.78, 0.96, 1.12) + vec3f(0.05, 0.2, 0.38) * 0.18;
  return liquid + vec3f(0.5, 0.9, 1.0) * (caustic * 0.12 * soft + edge * 0.18);
}

fn clothMaterial(uv: vec2f, soft: f32) -> vec3f {
  let t = params.time;
  let px = uv * params.resolution;
  let fold = sin(uv.y * 18.0 + t * 0.55) * sin(uv.x * 7.5 - t * 0.35);
  let warp = abs(sin(px.x * 1.14 + fold * 2.4));
  let weft = abs(sin(px.y * 1.28 - fold * 2.0));
  let crossThread = smoothstep(0.58, 0.96, warp) * 0.045 + smoothstep(0.58, 0.96, weft) * 0.04;
  let twill = sin((px.x + px.y) * 0.48 + fold * 1.8) * 0.035;
  let normal = maskNormal(uv);
  let refracted = sampleHiddenFill(clamp(uv + (normal.xy * 4.0 / params.resolution) * soft, vec2f(0.001), vec2f(0.999)));
  let shade = 0.78 + crossThread + twill + fold * 0.08;
  let fabric = mix(refracted, vec3f(0.6, 0.64, 0.6), 0.28) * shade;
  let edge = soft * (1.0 - soft) * 4.0;
  return fabric + vec3f(0.9, 0.95, 0.88) * edge * 0.08;
}

@fragment
fn fragmentMain(input: VertexOut) -> @location(0) vec4f {
  let uv = input.uv;
  let frame = textureSample(frameTex, linearSampler, uv).rgb;
  let mask = textureSample(refinedMaskTex, linearSampler, uv).r;
  let fill = sampleHiddenFill(uv);

  if (params.debugView > 0.5 && params.debugView < 1.5) {
    return vec4f(vec3f(mask), 1.0);
  }
  if (params.debugView > 1.5) {
    return vec4f(fill, 1.0);
  }

  let soft = smoothstep(0.02, 0.28, mask);
  var finalFill = fill;
  if (params.bodyFxMode > 0.5 && params.bodyFxMode < 1.5) {
    finalFill = jellyMaterial(uv, soft);
  } else if (params.bodyFxMode > 1.5 && params.bodyFxMode < 2.5) {
    finalFill = waterMaterial(uv, soft);
  } else if (params.bodyFxMode > 2.5) {
    finalFill = clothMaterial(uv, soft);
  }

  let blend = soft * params.opacity;
  var finalColor = mix(frame, finalFill, blend);

  let edge = soft * (1.0 - soft) * 4.0;
  finalColor = finalColor + vec3f(0.5, 0.85, 1.0) * edge * params.edgeGain;
  return vec4f(finalColor, 1.0);
}
