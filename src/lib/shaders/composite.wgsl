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

fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.299, 0.587, 0.114));
}

fn hash21(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2f) -> f32 {
  var v = 0.0;
  var a = 0.5;
  var q = p;
  for (var i = 0; i < 4; i = i + 1) {
    v = v + valueNoise(q) * a;
    q = q * 2.03 + vec2f(17.3, 11.7);
    a = a * 0.5;
  }
  return v;
}

fn sampleBodyFrame(uv: vec2f) -> vec3f {
  let px = 1.0 / params.resolution;
  var acc = textureSample(frameTex, linearSampler, uv).rgb * 4.0;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv + vec2f(px.x * 1.5, 0.0), vec2f(0.001), vec2f(0.999))).rgb;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv - vec2f(px.x * 1.5, 0.0), vec2f(0.001), vec2f(0.999))).rgb;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv + vec2f(0.0, px.y * 1.5), vec2f(0.001), vec2f(0.999))).rgb;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv - vec2f(0.0, px.y * 1.5), vec2f(0.001), vec2f(0.999))).rgb;
  return acc / 8.0;
}

fn sampleBodyVolume(uv: vec2f) -> vec3f {
  let px = 1.0 / params.resolution;
  var acc = textureSample(frameTex, linearSampler, uv).rgb * 5.0;
  var total = 5.0;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv + vec2f(px.x * 4.0, 0.0), vec2f(0.001), vec2f(0.999))).rgb * 2.0;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv - vec2f(px.x * 4.0, 0.0), vec2f(0.001), vec2f(0.999))).rgb * 2.0;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv + vec2f(0.0, px.y * 4.0), vec2f(0.001), vec2f(0.999))).rgb * 2.0;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv - vec2f(0.0, px.y * 4.0), vec2f(0.001), vec2f(0.999))).rgb * 2.0;
  total = total + 8.0;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv + vec2f(px.x * 7.0, px.y * 5.0), vec2f(0.001), vec2f(0.999))).rgb;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv + vec2f(-px.x * 7.0, px.y * 5.0), vec2f(0.001), vec2f(0.999))).rgb;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv + vec2f(px.x * 7.0, -px.y * 5.0), vec2f(0.001), vec2f(0.999))).rgb;
  acc = acc + textureSample(frameTex, linearSampler, clamp(uv + vec2f(-px.x * 7.0, -px.y * 5.0), vec2f(0.001), vec2f(0.999))).rgb;
  total = total + 4.0;
  return acc / total;
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

fn bodyNormal(uv: vec2f, soft: f32) -> vec3f {
  let px = 1.0 / params.resolution;
  let left = luminance(textureSample(frameTex, linearSampler, clamp(uv - vec2f(px.x * 2.0, 0.0), vec2f(0.001), vec2f(0.999))).rgb);
  let right = luminance(textureSample(frameTex, linearSampler, clamp(uv + vec2f(px.x * 2.0, 0.0), vec2f(0.001), vec2f(0.999))).rgb);
  let up = luminance(textureSample(frameTex, linearSampler, clamp(uv - vec2f(0.0, px.y * 2.0), vec2f(0.001), vec2f(0.999))).rgb);
  let down = luminance(textureSample(frameTex, linearSampler, clamp(uv + vec2f(0.0, px.y * 2.0), vec2f(0.001), vec2f(0.999))).rgb);
  let internal = vec2f(left - right, up - down) * 1.7;
  let edge = soft * (1.0 - soft) * 4.0;
  let silhouette = maskNormal(uv).xy * edge * 1.6;
  return normalize(vec3f(internal + silhouette, 1.0));
}

fn bodySpecular(normal: vec3f, power: f32) -> f32 {
  let light = normalize(vec3f(-0.35, -0.55, 0.9));
  let view = vec3f(0.0, 0.0, 1.0);
  let halfVector = normalize(light + view);
  return pow(max(dot(normal, halfVector), 0.0), power);
}

fn sampleBodyGuarded(uv: vec2f, fallbackUv: vec2f) -> vec3f {
  let mask = sampleMaskAt(uv, vec2f(0.0));
  let fallback = sampleBodyVolume(fallbackUv);
  let displaced = sampleBodyVolume(clamp(uv, vec2f(0.001), vec2f(0.999)));
  return mix(fallback, displaced, smoothstep(0.12, 0.55, mask));
}

fn waterWave(uv: vec2f, t: f32) -> f32 {
  let longWave = sin(uv.x * 34.0 + sin(uv.y * 8.0 + t * 0.9) * 2.4 - t * 2.6);
  let crossing = sin((uv.x * -18.0 + uv.y * 29.0) + t * 2.1);
  let small = fbm(uv * vec2f(18.0, 14.0) + vec2f(t * 0.42, -t * 0.31)) * 2.0 - 1.0;
  return longWave * 0.5 + crossing * 0.28 + small * 0.22;
}

fn waterGradient(uv: vec2f, t: f32) -> vec2f {
  let px = 2.0 / params.resolution;
  let dx = waterWave(uv + vec2f(px.x, 0.0), t) - waterWave(uv - vec2f(px.x, 0.0), t);
  let dy = waterWave(uv + vec2f(0.0, px.y), t) - waterWave(uv - vec2f(0.0, px.y), t);
  return vec2f(dx, dy);
}

fn jellyMaterial(uv: vec2f, soft: f32) -> vec3f {
  let t = params.time;
  let wobbleA = fbm(uv * vec2f(5.0, 4.0) + vec2f(t * 0.46, -t * 0.27)) - 0.5;
  let wobbleB = fbm(uv * vec2f(4.0, 6.0) + vec2f(-t * 0.34, t * 0.39)) - 0.5;
  let displacement = vec2f(wobbleA, wobbleB) * 8.0 + bodyNormal(uv, soft).xy * 2.0;
  let body = sampleBodyGuarded(uv + (displacement / params.resolution) * soft, uv);
  let edgeNormal = maskNormal(uv);
  let normal = normalize(vec3f(edgeNormal.xy * 0.45 + vec2f(wobbleA, wobbleB) * 0.42, 1.0));
  let edge = soft * (1.0 - soft) * 4.0;
  let volume = smoothstep(0.18, 0.9, soft);
  let spec = bodySpecular(normal, 46.0) * 0.3;
  let pulse = sin(t * 2.4 + uv.y * 13.0 + wobbleA * 4.0) * 0.5 + 0.5;
  var material = body * (0.9 + volume * 0.1 + pulse * 0.025);
  material = material + vec3f(1.0) * (edge * 0.12 + spec * volume);
  return material;
}

fn waterMaterial(uv: vec2f, soft: f32) -> vec3f {
  let t = params.time;
  let wave = waterWave(uv, t);
  let gradient = waterGradient(uv, t);
  let displacement = (gradient * 11.0 + maskNormal(uv).xy * 2.0) * soft;
  let body = sampleBodyGuarded(uv + displacement / params.resolution, uv);
  let normal = normalize(vec3f(gradient * 0.72 + maskNormal(uv).xy * 0.32, 1.0));
  let edge = soft * (1.0 - soft) * 4.0;
  let waveBand = abs(sin(uv.x * 38.0 + sin(uv.y * 9.0 + t * 0.7) * 2.2 - t * 2.8));
  let rippleLine = smoothstep(0.9, 0.985, waveBand) * (0.45 + 0.55 * smoothstep(-0.2, 0.8, wave));
  let spec = bodySpecular(normal, 78.0) * 0.28;
  var material = body * (0.86 + wave * 0.035);
  material = material + vec3f(1.0) * (rippleLine * 0.2 * soft + spec + edge * 0.12);
  material = material - vec3f(1.0) * (1.0 - soft) * 0.03;
  return material;
}

fn clothMaterial(uv: vec2f, soft: f32) -> vec3f {
  let t = params.time;
  let px = uv * params.resolution;
  let body = sampleBodyFrame(uv);
  let baseLuma = smoothstep(0.04, 0.95, luminance(body));
  let fold = sin(uv.y * 18.0 + t * 0.55) * sin(uv.x * 7.5 - t * 0.35);
  let warp = abs(sin(px.x * 1.14 + fold * 2.4));
  let weft = abs(sin(px.y * 1.28 - fold * 2.0));
  let crossThread = smoothstep(0.58, 0.96, warp) * 0.045 + smoothstep(0.58, 0.96, weft) * 0.04;
  let twill = sin((px.x + px.y) * 0.48 + fold * 1.8) * 0.035;
  let shade = 0.78 + baseLuma * 0.18 + crossThread + twill + fold * 0.08;
  let fabric = body * shade;
  let edge = soft * (1.0 - soft) * 4.0;
  return fabric + vec3f(1.0) * edge * 0.06;
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
