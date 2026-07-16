export {
  buildLowFrequencyTrainingPoses,
  selectCubatureSamples,
  type CubatureSelection,
} from "./cubature";
export {
  assembleRestLinearSystem,
  computeLinearTetrahedronStiffness,
  computeLumpedMasses,
  computeRestTetraData,
  type TetrahedronStiffnessResult,
} from "./fem";
export {
  choleskyFactor,
  determinant3,
  dot,
  invert3,
  multiply3,
  solveCholesky,
  solveDenseLinearSystem,
  squaredNorm,
} from "./math";
export {
  appendTetrahedralMeshes,
  extractBoundarySurface,
  generateRegularCuboidMesh,
  getTetrahedronCount,
  getVertexCount,
  transformTetrahedralMesh,
  validateTetrahedralMesh,
} from "./mesh";
export {
  buildPrecomputedScene,
  computeExactVertexBasis,
  exactBasisToActiveMatrix,
} from "./precompute";
export type {
  ColorRgba,
  CubatureSample,
  LinearMaterial,
  MeshTransform,
  PrecomputedScene,
  PrecomputeOptions,
  RegularCuboidOptions,
  RestLinearSystem,
  RestTetraData,
  SceneCamera,
  SceneDefinition,
  SceneLandmark,
  SimulationSettings,
  SurfaceTopology,
  TetrahedralMesh,
  Vec3,
  VertexPrecomputation,
} from "./types";
