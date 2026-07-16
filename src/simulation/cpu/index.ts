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
  centralDifferenceGradient,
  centralDifferenceHessian,
  relativeError,
  type ScalarFunction,
  type VectorFunction,
} from "./finite-difference";
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
export {
  activeCoordinatesFromFullPositions,
  computeCorotatedLinearRotations,
  computeDirectComplementaryEquilibriumBasis,
  createCorotatedLinearImplicitEulerOracle,
  evaluateCorotatedLinearTetrahedron,
  fullPositionsFromActiveCoordinates,
  polarRotation3,
  solveEquilibriumBasisNewtonStep,
  solveFullNewtonStep,
  type ComplementaryEquilibriumBasis,
  type CorotatedLinearElementEvaluation,
  type CorotatedLinearImplicitEulerOptions,
  type CorotatedLinearImplicitEulerOracle,
  type EnergyEvaluation,
  type ImplicitEulerEnergyComponents,
  type ImplicitEulerEvaluation,
  type QuadraticFloorContact,
} from "./oracle";
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
