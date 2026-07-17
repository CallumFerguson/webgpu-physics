export type Vec3 = readonly [number, number, number];
export type ColorRgba = readonly [number, number, number, number];
export type ElasticMaterialModel =
  | "corotated-linear"
  | "stable-neo-hookean";

export interface LinearMaterial {
  readonly name: string;
  /** Defaults to the Phase 0 co-rotated regression material. */
  readonly model?: ElasticMaterialModel;
  readonly density: number;
  readonly youngModulus: number;
  readonly poissonRatio: number;
  readonly color: ColorRgba;
}

export interface TetrahedralMesh {
  /** Rest positions, tightly packed xyz per vertex. */
  readonly positions: Float64Array;
  /** Four vertex indices per positively oriented tetrahedron. */
  readonly tetrahedra: Uint32Array;
  /** One material index per tetrahedron. */
  readonly materialIds: Uint16Array;
  /** Non-zero entries identify kinematic vertices. */
  readonly fixed: Uint8Array;
  /** Connected-body identifier used by collision/rendering code. */
  readonly bodyIds: Uint16Array;
}

export interface SurfaceTopology {
  /** Outward-oriented boundary triangles. */
  readonly triangles: Uint32Array;
  /** Unique undirected boundary edges, stored as sorted vertex pairs. */
  readonly edges: Uint32Array;
}

export interface RestTetraData {
  readonly volumes: Float64Array;
  /** Row-major inverse rest matrices, nine values per tetrahedron. */
  readonly inverseRestMatrices: Float64Array;
  /** Row-major 12 by 12 linear elastic Hessians. */
  readonly stiffnessMatrices: Float64Array;
}

export interface RestLinearSystem {
  readonly dimension: number;
  readonly activeVertices: Uint32Array;
  /** Base active DOF for a vertex, or -1 for a fixed vertex. */
  readonly vertexToActiveDof: Int32Array;
  /** Dense row-major active-coordinate Hessian. */
  readonly hessian: Float64Array;
  /** Dense row-major lower Cholesky factor of hessian. */
  readonly choleskyLower: Float64Array;
}

export interface CubatureSample {
  readonly tetrahedron: number;
  readonly weight: number;
  /** Four row-major 3 by 3 blocks of the rest perturbation basis. */
  readonly basisBlocks: Float64Array;
}

export interface VertexPrecomputation {
  readonly vertex: number;
  /**
   * Dense full-coordinate Ubar_i. Each vertex owns one row-major 3 by 3
   * block, so the array has vertexCount * 9 entries.
   */
  readonly exactBasis?: Float64Array;
  /** (S_i H^-1 S_i^T)^-1, row-major 3 by 3. */
  readonly schurInverse: Float64Array;
  readonly cubature: readonly CubatureSample[];
  /** Explicit when preprocessing selected a material-specific trainer. */
  readonly cubatureModel?: ElasticMaterialModel;
  /** Float64 fit before runtime f32 packing, when separately measured. */
  readonly trainingResidualF64?: number;
  /** Runtime-packed residual for nonlinear Cubature; legacy f64 residual otherwise. */
  readonly trainingResidual: number;
  readonly validTrainingPoseCount?: number;
  readonly trivialTrainingPoseCount?: number;
  readonly nonzeroTrainingCandidateCount?: number;
  readonly trainingColumnRank?: number;
  readonly packedNonzeroTrainingCandidateCount?: number;
  readonly packedTrainingColumnRank?: number;
}

export interface SimulationSettings {
  readonly timestep: number;
  readonly gravity: Vec3;
  readonly floorY: number;
  readonly solverIterations: number;
  readonly cubatureSamples: 4 | 6;
  /** Initial linear velocity indexed by body id; omitted body entries are zero. */
  readonly initialBodyVelocities?: readonly Vec3[];
}

/** One deliberately small triangle-cloth layer sharing the scene vertex array. */
export interface TriangleClothDefinition {
  /** Three consistently oriented vertex indices per membrane triangle. */
  readonly triangles: Uint32Array;
  /** Volumetric density; areal mass is density times thickness. */
  readonly density: number;
  readonly youngModulus: number;
  readonly poissonRatio: number;
  readonly thickness: number;
  /** Rest-edge-weighted quadratic dihedral stiffness. */
  readonly bendingStiffness: number;
  readonly color: ColorRgba;
}

export interface SceneDefinition {
  readonly id: string;
  readonly title: string;
  readonly description: string;
  readonly mesh: TetrahedralMesh;
  readonly cloth?: TriangleClothDefinition;
  readonly materials: readonly LinearMaterial[];
  readonly settings: SimulationSettings;
  readonly camera: SceneCamera;
  readonly landmark: SceneLandmark;
}

export interface PrecomputedScene extends SceneDefinition {
  readonly surface: SurfaceTopology;
  readonly restTetraData: RestTetraData;
  readonly lumpedMasses: Float64Array;
  readonly restSystem: RestLinearSystem;
  readonly vertexPrecomputations: readonly VertexPrecomputation[];
}

export interface RegularCuboidOptions {
  readonly cells: readonly [number, number, number];
  readonly origin?: Vec3;
  readonly size?: Vec3;
  readonly materialId?: number;
  readonly bodyId?: number;
  readonly fixed?: (
    position: Vec3,
    gridCoordinate: readonly [number, number, number],
  ) => boolean;
}

export interface MeshTransform {
  readonly translation?: Vec3;
  readonly scale?: number | Vec3;
  /** Intrinsic XYZ Euler rotation, in radians. */
  readonly rotationEuler?: Vec3;
}

export interface SceneCamera {
  readonly eye: Vec3;
  readonly target: Vec3;
  readonly up: Vec3;
  readonly fovYRadians: number;
}

export interface SceneLandmark {
  readonly vertex: number;
  readonly label: string;
  readonly expectedMotion: Vec3;
}

export interface PrecomputeOptions {
  /** Retain dense exact bases for diagnostics and mathematical unit tests. */
  readonly retainExactBases?: boolean;
}
