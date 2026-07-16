import type { PrecomputedScene } from "../scenes";

export type DiagnosticVec3 = readonly [number, number, number];

export interface JGS2BodyDiagnostics {
  readonly bodyId: number;
  readonly mass: number;
  readonly centerOfMass: DiagnosticVec3;
  readonly linearVelocity: DiagnosticVec3;
  readonly linearMomentum: DiagnosticVec3;
  /** Angular momentum about this body's center of mass. */
  readonly angularMomentum: DiagnosticVec3;
  readonly minY: number;
}

export interface JGS2DiagnosticRuntimeSettings {
  readonly parityMode: boolean;
  readonly velocityDamping: number;
  readonly contactTangentialDamping: number;
  readonly horizontalBodyCorrection: boolean;
}

export interface JGS2Diagnostics {
  readonly frame: number;
  readonly timestep: number;
  readonly lastStepIterations: number;
  /** All metrics in this object are derived on the CPU after explicit readback. */
  readonly source: "cpu-readback";
  readonly runtime: JGS2DiagnosticRuntimeSettings;
  readonly finite: boolean;
  readonly pinnedMaxError: number;
  readonly minTetDeterminant: number;
  readonly minTetDeterminantValid: boolean;
  /** Finite sentinel until the general contact pipeline supplies this reduction. */
  readonly minimumContactDistance: number;
  readonly minimumContactDistanceValid: false;
  /** Finite sentinel until the general contact pipeline supplies this reduction. */
  readonly activeContactCount: number;
  readonly activeContactCountValid: false;
  /** False sentinel until a bounded contact-candidate buffer exists. */
  readonly candidateBufferOverflow: boolean;
  readonly candidateBufferOverflowValid: false;
  /** Finite sentinel; no runtime residual reduction exists yet. */
  readonly relativeResidual: number;
  readonly relativeResidualValid: false;
  /** Finite sentinel; no runtime maximum-update reduction exists yet. */
  readonly maximumUpdate: number;
  readonly maximumUpdateValid: false;
  readonly totalLinearMomentum: DiagnosticVec3;
  readonly totalLinearMomentumValid: boolean;
  /** Angular momentum about the world origin. */
  readonly totalAngularMomentum: DiagnosticVec3;
  readonly totalAngularMomentumValid: boolean;
  readonly floorHeight: number;
  readonly bounds: {
    readonly min: DiagnosticVec3;
    readonly max: DiagnosticVec3;
  };
  readonly landmark: DiagnosticVec3;
  readonly comparisonLandmark?: DiagnosticVec3;
  readonly bodies: readonly JGS2BodyDiagnostics[];
}

export interface JGS2DiagnosticsContext {
  readonly frame: number;
  readonly lastStepIterations: number;
  readonly runtime: JGS2DiagnosticRuntimeSettings;
}

function determinant(
  positions: Float32Array,
  a: number,
  b: number,
  c: number,
  d: number,
): number {
  const ax = positions[a * 4] ?? 0;
  const ay = positions[a * 4 + 1] ?? 0;
  const az = positions[a * 4 + 2] ?? 0;
  const bx = (positions[b * 4] ?? 0) - ax;
  const by = (positions[b * 4 + 1] ?? 0) - ay;
  const bz = (positions[b * 4 + 2] ?? 0) - az;
  const cx = (positions[c * 4] ?? 0) - ax;
  const cy = (positions[c * 4 + 1] ?? 0) - ay;
  const cz = (positions[c * 4 + 2] ?? 0) - az;
  const dx = (positions[d * 4] ?? 0) - ax;
  const dy = (positions[d * 4 + 1] ?? 0) - ay;
  const dz = (positions[d * 4 + 2] ?? 0) - az;
  return (
    bx * (cy * dz - cz * dy) -
    cx * (by * dz - bz * dy) +
    dx * (by * cz - bz * cy)
  );
}

function inferBodyCount(bodyIds: ArrayLike<number>): number {
  let maximumBodyId = 0;
  for (let index = 0; index < bodyIds.length; index += 1) {
    const bodyId = bodyIds[index] ?? 0;
    maximumBodyId = Math.max(maximumBodyId, bodyId);
  }
  return bodyIds.length === 0 ? 0 : maximumBodyId + 1;
}

function addCrossProduct(
  output: Float64Array,
  outputOffset: number,
  ax: number,
  ay: number,
  az: number,
  bx: number,
  by: number,
  bz: number,
): void {
  output[outputOffset] =
    (output[outputOffset] ?? 0) + ay * bz - az * by;
  output[outputOffset + 1] =
    (output[outputOffset + 1] ?? 0) + az * bx - ax * bz;
  output[outputOffset + 2] =
    (output[outputOffset + 2] ?? 0) + ax * by - ay * bx;
}

function vec3(values: Float64Array, offset = 0): DiagnosticVec3 {
  return [
    values[offset] ?? 0,
    values[offset + 1] ?? 0,
    values[offset + 2] ?? 0,
  ];
}

function finiteVec3(value: DiagnosticVec3): boolean {
  return value.every(Number.isFinite);
}

export function diagnosticsFromState(
  scene: PrecomputedScene,
  positions: Float32Array,
  velocities: Float32Array,
  context: JGS2DiagnosticsContext,
): JGS2Diagnostics {
  const vertexCount = positions.length / 4;
  if (!Number.isSafeInteger(vertexCount) || velocities.length !== positions.length) {
    throw new RangeError(
      "Diagnostic position and velocity arrays must contain matching vec4 values.",
    );
  }

  const minimum: [number, number, number] = [Infinity, Infinity, Infinity];
  const maximum: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  let finite = true;
  let pinnedMaxError = 0;
  const bodyCount = inferBodyCount(scene.mesh.bodyIds);
  const bodyMasses = new Float64Array(bodyCount);
  const bodyPositionMoments = new Float64Array(bodyCount * 3);
  const bodyVelocityMoments = new Float64Array(bodyCount * 3);
  const bodyAngularMomenta = new Float64Array(bodyCount * 3);
  const bodyMinimumY = new Float64Array(bodyCount);
  const totalLinearMomentumValues = new Float64Array(3);
  const totalAngularMomentumValues = new Float64Array(3);
  bodyMinimumY.fill(Infinity);

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const body = scene.mesh.bodyIds[vertex] ?? 0;
    const mass = scene.lumpedMasses[vertex] ?? 0;
    bodyMasses[body] += mass;
    const positionOffset = vertex * 4;
    const px = positions[positionOffset] ?? Number.NaN;
    const py = positions[positionOffset + 1] ?? Number.NaN;
    const pz = positions[positionOffset + 2] ?? Number.NaN;
    const vx = velocities[positionOffset] ?? Number.NaN;
    const vy = velocities[positionOffset + 1] ?? Number.NaN;
    const vz = velocities[positionOffset + 2] ?? Number.NaN;
    const momentumX = mass * vx;
    const momentumY = mass * vy;
    const momentumZ = mass * vz;

    finite &&= [mass, px, py, pz, vx, vy, vz].every(Number.isFinite);
    minimum[0] = Math.min(minimum[0], px);
    minimum[1] = Math.min(minimum[1], py);
    minimum[2] = Math.min(minimum[2], pz);
    maximum[0] = Math.max(maximum[0], px);
    maximum[1] = Math.max(maximum[1], py);
    maximum[2] = Math.max(maximum[2], pz);

    bodyPositionMoments[body * 3] += mass * px;
    bodyPositionMoments[body * 3 + 1] += mass * py;
    bodyPositionMoments[body * 3 + 2] += mass * pz;
    bodyVelocityMoments[body * 3] += momentumX;
    bodyVelocityMoments[body * 3 + 1] += momentumY;
    bodyVelocityMoments[body * 3 + 2] += momentumZ;
    totalLinearMomentumValues[0] += momentumX;
    totalLinearMomentumValues[1] += momentumY;
    totalLinearMomentumValues[2] += momentumZ;
    addCrossProduct(
      totalAngularMomentumValues,
      0,
      px,
      py,
      pz,
      momentumX,
      momentumY,
      momentumZ,
    );

    if (scene.mesh.fixed[vertex] !== 0) {
      pinnedMaxError = Math.max(
        pinnedMaxError,
        Math.abs(px - (scene.mesh.positions[vertex * 3] ?? 0)),
        Math.abs(py - (scene.mesh.positions[vertex * 3 + 1] ?? 0)),
        Math.abs(pz - (scene.mesh.positions[vertex * 3 + 2] ?? 0)),
      );
    }
    bodyMinimumY[body] = Math.min(bodyMinimumY[body] ?? Infinity, py);
  }

  for (let vertex = 0; vertex < vertexCount; vertex += 1) {
    const body = scene.mesh.bodyIds[vertex] ?? 0;
    const mass = bodyMasses[body] ?? 0;
    if (!(mass > 0)) {
      continue;
    }
    const vertexMass = scene.lumpedMasses[vertex] ?? 0;
    const positionOffset = vertex * 4;
    const bodyOffset = body * 3;
    const centerX = (bodyPositionMoments[bodyOffset] ?? 0) / mass;
    const centerY = (bodyPositionMoments[bodyOffset + 1] ?? 0) / mass;
    const centerZ = (bodyPositionMoments[bodyOffset + 2] ?? 0) / mass;
    const centerVelocityX = (bodyVelocityMoments[bodyOffset] ?? 0) / mass;
    const centerVelocityY = (bodyVelocityMoments[bodyOffset + 1] ?? 0) / mass;
    const centerVelocityZ = (bodyVelocityMoments[bodyOffset + 2] ?? 0) / mass;
    addCrossProduct(
      bodyAngularMomenta,
      bodyOffset,
      (positions[positionOffset] ?? 0) - centerX,
      (positions[positionOffset + 1] ?? 0) - centerY,
      (positions[positionOffset + 2] ?? 0) - centerZ,
      vertexMass * ((velocities[positionOffset] ?? 0) - centerVelocityX),
      vertexMass * ((velocities[positionOffset + 1] ?? 0) - centerVelocityY),
      vertexMass * ((velocities[positionOffset + 2] ?? 0) - centerVelocityZ),
    );
  }

  const hasTetrahedra = scene.mesh.tetrahedra.length >= 4;
  let minTetDeterminant = hasTetrahedra ? Infinity : 0;
  for (let offset = 0; offset < scene.mesh.tetrahedra.length; offset += 4) {
    minTetDeterminant = Math.min(
      minTetDeterminant,
      determinant(
        positions,
        scene.mesh.tetrahedra[offset] ?? 0,
        scene.mesh.tetrahedra[offset + 1] ?? 0,
        scene.mesh.tetrahedra[offset + 2] ?? 0,
        scene.mesh.tetrahedra[offset + 3] ?? 0,
      ),
    );
  }
  finite &&= !hasTetrahedra || Number.isFinite(minTetDeterminant);

  const landmarkVertex = scene.landmark.vertex;
  const landmark: DiagnosticVec3 = [
    positions[landmarkVertex * 4] ?? Number.NaN,
    positions[landmarkVertex * 4 + 1] ?? Number.NaN,
    positions[landmarkVertex * 4 + 2] ?? Number.NaN,
  ];
  const comparisonVertex =
    scene.id === "stiffness" ? landmarkVertex + vertexCount / 2 : -1;
  const comparisonLandmark: DiagnosticVec3 | undefined = comparisonVertex >= 0
    ? [
        positions[comparisonVertex * 4] ?? Number.NaN,
        positions[comparisonVertex * 4 + 1] ?? Number.NaN,
        positions[comparisonVertex * 4 + 2] ?? Number.NaN,
      ]
    : undefined;

  const bodies: JGS2BodyDiagnostics[] = [];
  for (let bodyId = 0; bodyId < bodyCount; bodyId += 1) {
    const mass = bodyMasses[bodyId] ?? 0;
    if (!(mass > 0)) {
      continue;
    }
    const offset = bodyId * 3;
    const linearMomentum = vec3(bodyVelocityMoments, offset);
    const body: JGS2BodyDiagnostics = {
      bodyId,
      mass,
      centerOfMass: [
        (bodyPositionMoments[offset] ?? 0) / mass,
        (bodyPositionMoments[offset + 1] ?? 0) / mass,
        (bodyPositionMoments[offset + 2] ?? 0) / mass,
      ],
      linearVelocity: [
        linearMomentum[0] / mass,
        linearMomentum[1] / mass,
        linearMomentum[2] / mass,
      ],
      linearMomentum,
      angularMomentum: vec3(bodyAngularMomenta, offset),
      minY: bodyMinimumY[bodyId] ?? Number.NaN,
    };
    finite &&=
      Number.isFinite(body.mass) &&
      finiteVec3(body.centerOfMass) &&
      finiteVec3(body.linearVelocity) &&
      finiteVec3(body.linearMomentum) &&
      finiteVec3(body.angularMomentum) &&
      Number.isFinite(body.minY);
    bodies.push(body);
  }

  const totalLinearMomentum = vec3(totalLinearMomentumValues);
  const totalAngularMomentum = vec3(totalAngularMomentumValues);
  const totalMass = bodyMasses.reduce((sum, mass) => sum + mass, 0);
  finite &&=
    finiteVec3(totalLinearMomentum) &&
    finiteVec3(totalAngularMomentum) &&
    finiteVec3(landmark) &&
    (!comparisonLandmark || finiteVec3(comparisonLandmark));

  return {
    frame: context.frame,
    timestep: scene.settings.timestep,
    lastStepIterations: context.lastStepIterations,
    source: "cpu-readback",
    runtime: context.runtime,
    finite,
    pinnedMaxError,
    minTetDeterminant,
    minTetDeterminantValid: hasTetrahedra,
    minimumContactDistance: 0,
    minimumContactDistanceValid: false,
    activeContactCount: 0,
    activeContactCountValid: false,
    candidateBufferOverflow: false,
    candidateBufferOverflowValid: false,
    relativeResidual: 0,
    relativeResidualValid: false,
    maximumUpdate: 0,
    maximumUpdateValid: false,
    totalLinearMomentum,
    totalLinearMomentumValid: totalMass > 0,
    totalAngularMomentum,
    totalAngularMomentumValid: totalMass > 0,
    floorHeight: scene.settings.floorY,
    bounds: { min: minimum, max: maximum },
    landmark,
    ...(comparisonLandmark ? { comparisonLandmark } : {}),
    bodies,
  };
}
