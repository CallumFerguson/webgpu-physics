export const triangleShader = /* wgsl */ `
struct VertexOutput {
  @builtin(position) position: vec4f,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  let positions = array<vec2f, 3>(
    vec2f(-0.72, -0.64),
    vec2f(0.72, -0.64),
    vec2f(0.0, 0.72),
  );

  var output: VertexOutput;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  return output;
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
  return vec4f(0.20, 0.86, 0.72, 1.0);
}
`;
