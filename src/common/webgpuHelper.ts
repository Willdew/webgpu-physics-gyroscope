// Webgpu config functions
export async function configCanvas(canvas: HTMLCanvasElement) {
  let context = canvas.getContext("webgpu")
  if (!context) {
    fail("Failed to get context")
  }

  const device = await getGpuDevice()
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat()
  context?.configure({
    device: device,
    format: presentationFormat,
    alphaMode: "opaque",
  })

  return {
    device: device,
    context: context,
    format: presentationFormat,
  }
}

export function createDepthStencil(canvas: HTMLCanvasElement, device: GPUDevice) {
  let depthTexture = device.createTexture({
    size: { width: canvas.width, height: canvas.height },
    format: "depth24plus",
    sampleCount: 4,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  return depthTexture
}

export function createMSAATexture(canvas: HTMLCanvasElement, device: GPUDevice, format: GPUTextureFormat) {
  let msaaTexture = device.createTexture({
    size: { width: canvas.width, height: canvas.height },
    format: format,
    sampleCount: 4,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  return msaaTexture
}

async function getGpuDevice(): Promise<GPUDevice> {
  let adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice()
  if (!device) {
    fail("Browser does not support WebGPU")
  }

  return device
}

export function fail(msg: string): never {
  document.body.innerHTML = `<h1>${msg}</h1>`;
  throw new Error(msg);
}


// Shape generators
export function createCuboidVertices(x: number, y: number, z: number): Float32Array {
  const xSize = x / 2;
  const ySize = y / 2;
  const zSize = z / 2;

  // Define corners
  const ftr = [xSize, ySize, zSize, 1.0]; // Front Top Right
  const ftl = [-xSize, ySize, zSize, 1.0]; // Front Top Left
  const fbl = [-xSize, -ySize, zSize, 1.0]; // Front Bottom Left
  const fbr = [xSize, -ySize, zSize, 1.0]; // Front Bottom Right

  const btr = [xSize, ySize, -zSize, 1.0]; // Back Top Right
  const btl = [-xSize, ySize, -zSize, 1.0]; // Back Top Left
  const bbl = [-xSize, -ySize, -zSize, 1.0]; // Back Bottom Left
  const bbr = [xSize, -ySize, -zSize, 1.0]; // Back Bottom Right

  // Face normals
  const frontNormal = [0, 0, 1, 0];
  const backNormal = [0, 0, -1, 0];
  const leftNormal = [-1, 0, 0, 0];
  const rightNormal = [1, 0, 0, 0];
  const topNormal = [0, 1, 0, 0];
  const bottomNormal = [0, -1, 0, 0];

  // Helper
  function vtx(pos: number[], normal: number[]) {
    return [...pos, ...normal];
  }

  const vertices = [
    // Front face
    ...vtx(ftl, frontNormal), ...vtx(ftr, frontNormal), ...vtx(fbr, frontNormal),
    ...vtx(ftl, frontNormal), ...vtx(fbr, frontNormal), ...vtx(fbl, frontNormal),

    // Back face
    ...vtx(btl, backNormal), ...vtx(btr, backNormal), ...vtx(bbr, backNormal),
    ...vtx(btl, backNormal), ...vtx(bbr, backNormal), ...vtx(bbl, backNormal),

    // Left face
    ...vtx(btl, leftNormal), ...vtx(ftl, leftNormal), ...vtx(fbl, leftNormal),
    ...vtx(btl, leftNormal), ...vtx(fbl, leftNormal), ...vtx(bbl, leftNormal),

    // Right face
    ...vtx(ftr, rightNormal), ...vtx(btr, rightNormal), ...vtx(bbr, rightNormal),
    ...vtx(ftr, rightNormal), ...vtx(bbr, rightNormal), ...vtx(fbr, rightNormal),

    // Top face
    ...vtx(ftl, topNormal), ...vtx(btl, topNormal), ...vtx(btr, topNormal),
    ...vtx(ftl, topNormal), ...vtx(btr, topNormal), ...vtx(ftr, topNormal),

    // Bottom face
    ...vtx(fbr, bottomNormal), ...vtx(bbr, bottomNormal), ...vtx(bbl, bottomNormal),
    ...vtx(fbr, bottomNormal), ...vtx(bbl, bottomNormal), ...vtx(fbl, bottomNormal),
  ];

  return new Float32Array(vertices);
}

export function createSphereVertices(subdivisions: number): Float32Array {
  const latSegments = subdivisions;
  const lonSegments = subdivisions;
  const radius = 1.0;

  const vertices: number[] = [];

  // Precompute positions
  const positions: [number, number, number, number][][] = [];
  for (let lat = 0; lat <= latSegments; lat++) {
    const phi = (lat / latSegments) * Math.PI;
    positions[lat] = [];
    for (let lon = 0; lon <= lonSegments; lon++) {
      const theta = (lon / lonSegments) * 2.0 * Math.PI;
      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.cos(phi);
      const z = radius * Math.sin(phi) * Math.sin(theta);
      positions[lat][lon] = [x, y, z, 1.0];
    }
  }

  function vtx(pos: number[]) {
    const nx = pos[0];
    const ny = pos[1];
    const nz = pos[2];
    return [...pos, nx, ny, nz, 0.0];
  }

  // Create triangles
  for (let lat = 0; lat < latSegments; lat++) {
    for (let lon = 0; lon < lonSegments; lon++) {
      const v1 = positions[lat][lon];
      const v2 = positions[lat + 1][lon];
      const v3 = positions[lat + 1][lon + 1];
      const v4 = positions[lat][lon + 1];

      // Triangle 1: v1, v2, v3
      vertices.push(...vtx(v1), ...vtx(v2), ...vtx(v3));
      // Triangle 2: v1, v3, v4
      vertices.push(...vtx(v1), ...vtx(v3), ...vtx(v4));
    }
  }

  return new Float32Array(vertices);
}
