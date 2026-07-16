export const sceneShader = /* wgsl */ `
struct RenderUniforms {
  viewProjection: mat4x4f,
  cameraAndTime: vec4f,
  floorAndScale: vec4f,
  counts: vec4u,
}

struct RenderVertex {
  rest: vec4f,
  color: vec4f,
}

struct VertexOutput {
  @builtin(position) clipPosition: vec4f,
  @location(0) worldPosition: vec3f,
  @location(1) color: vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(1) var<storage, read> dynamics: array<vec4f>;
@group(0) @binding(2) var<storage, read> vertices: array<RenderVertex>;

@vertex
fn liveVertex(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let position = dynamics[uniforms.counts.x + vertexIndex].xyz;
  var output: VertexOutput;
  output.clipPosition = uniforms.viewProjection * vec4f(position, 1.0);
  output.worldPosition = position;
  output.color = vertices[vertexIndex].color;
  return output;
}

@vertex
fn restVertex(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let position = vertices[vertexIndex].rest.xyz;
  var output: VertexOutput;
  output.clipPosition = uniforms.viewProjection * vec4f(position, 1.0);
  output.worldPosition = position;
  output.color = vec4f(0.25, 0.92, 1.0, 0.42);
  return output;
}

@fragment
fn surfaceFragment(input: VertexOutput) -> @location(0) vec4f {
  var normal = normalize(cross(dpdx(input.worldPosition), dpdy(input.worldPosition)));
  if (dot(normal, uniforms.cameraAndTime.xyz - input.worldPosition) < 0.0) {
    normal = -normal;
  }
  let lightDirection = normalize(vec3f(-0.35, 0.82, 0.48));
  let diffuse = 0.28 + 0.72 * max(dot(normal, lightDirection), 0.0);
  let rim = pow(1.0 - max(dot(normal, normalize(uniforms.cameraAndTime.xyz - input.worldPosition)), 0.0), 2.0);
  let color = input.color.rgb * diffuse + vec3f(0.16, 0.27, 0.38) * rim;
  return vec4f(color, 1.0);
}

@fragment
fn lineFragment(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}

@vertex
fn floorVertex(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let corner = corners[vertexIndex];
  let center = uniforms.floorAndScale.yz;
  let scale = uniforms.floorAndScale.w;
  let position = vec3f(center.x + corner.x * scale, uniforms.floorAndScale.x, center.y + corner.y * scale);
  var output: VertexOutput;
  output.clipPosition = uniforms.viewProjection * vec4f(position, 1.0);
  output.worldPosition = position;
  output.color = vec4f(1.0);
  return output;
}

@fragment
fn floorFragment(input: VertexOutput) -> @location(0) vec4f {
  let spacing = max(uniforms.floorAndScale.w / 18.0, 0.025);
  let gridCoordinate = abs(fract(input.worldPosition.xz / spacing + 0.5) - 0.5) / fwidth(input.worldPosition.xz / spacing);
  let gridLine = 1.0 - min(min(gridCoordinate.x, gridCoordinate.y), 1.0);
  let majorCoordinate = abs(fract(input.worldPosition.xz / (spacing * 5.0) + 0.5) - 0.5) / fwidth(input.worldPosition.xz / (spacing * 5.0));
  let majorLine = 1.0 - min(min(majorCoordinate.x, majorCoordinate.y), 1.0);
  let radialFade = 1.0 - smoothstep(uniforms.floorAndScale.w * 0.25, uniforms.floorAndScale.w, distance(input.worldPosition.xz, uniforms.floorAndScale.yz));
  let base = vec3f(0.025, 0.045, 0.072);
  let grid = vec3f(0.10, 0.22, 0.29) * gridLine + vec3f(0.10, 0.43, 0.48) * majorLine;
  return vec4f((base + grid * 0.55) * (0.55 + 0.45 * radialFade), 1.0);
}
`;
