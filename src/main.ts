import './style.css'
import vert from './shaders/shader.wgsl?raw'
import { mat4, vec3 } from 'wgpu-matrix'
import * as webgpuHelper from './common/webgpuHelper.ts'
import RAPIER from '@dimforge/rapier3d-compat'
await RAPIER.init();

const canvas = document.getElementById("canvas") as HTMLCanvasElement
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Global GPU state
let device: GPUDevice;
let context: GPUCanvasContext;
let format: GPUTextureFormat;
let pipeline: GPURenderPipeline;
let groundBuffer: GPUBuffer;
let sphereBuffer: GPUBuffer;
let groundUniformBuffer: GPUBuffer;
let sphereUniformBuffer: GPUBuffer;
let lightUniformBuffer: GPUBuffer;
let groundBindGroup: GPUBindGroup;
let sphereBindGroup: GPUBindGroup;
let depthStencil: GPUTexture;
let msaaTexture: GPUTexture;
let world: any;
let sphereRigidBody: any;

// Scene data
let sphereCoordinates = vec3.fromValues(0, 10, 0)

const [gX, gY, gZ] = [20, 0.01, 20]
const ground = webgpuHelper.createCuboidVertices(gX, gY, gZ);
const sphere = webgpuHelper.createSphereVertices(32);

// Camera and projection
const fov = 60 * Math.PI / 180;
let aspect = 1.0; // Will set after canvas is known
const near = 0.1;
const far = 1000;
let perspective: any;
const eye = [0, 10, 20];
const target = [0, 4, 0];
const up = [0, 1, 0];
let view = mat4.lookAt(eye, target, up);

// Models
let groundModelMatrix = mat4.rotationZ(0.0);

// Initialize everything and start rendering
init().then(() => {
  requestAnimationFrame(render);
});

// ------------------ FUNCTIONS ------------------

// Initialization logic
async function init() {
  ({ device, context, format } = await webgpuHelper.configCanvas(canvas));
  aspect = canvas.width / canvas.height;
  perspective = mat4.perspective(fov, aspect, near, far);

  depthStencil = webgpuHelper.createDepthStencil(canvas, device);
  msaaTexture = webgpuHelper.createMSAATexture(canvas, device, format);

  groundBuffer = createVertexBuffer(device, ground);
  sphereBuffer = createVertexBuffer(device, sphere);

  pipeline = createPipeline(device, format);

  const { groundUBO, sphereUBO, lightUBO } = createUniformBuffers(device);
  groundUniformBuffer = groundUBO;
  sphereUniformBuffer = sphereUBO;
  lightUniformBuffer = lightUBO;
  initRapier();

  ({ groundBindGroup, sphereBindGroup } = createBindGroups(device, pipeline, groundUniformBuffer, sphereUniformBuffer, lightUniformBuffer));

  updateUniforms(); // Set initial uniforms
}

function createVertexBuffer(device: GPUDevice, data: Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function createPipeline(device: GPUDevice, format: GPUTextureFormat): GPURenderPipeline {
  return device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: vert }),
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 32, // 8 floats * 4 bytes = 32
          attributes: [
            { shaderLocation: 0, format: "float32x4", offset: 0 },   // position
            { shaderLocation: 1, format: "float32x4", offset: 16 },  // normal
          ]
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({ code: vert }),
      entryPoint: "fs_main",
      targets: [{ format }]
    },
    primitive: {
      topology: "triangle-list"
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less"
    },
    multisample: {
      count: 4, // matching MSAA texture
    }
  });
}

function createUniformBuffers(device: GPUDevice) {
  const uniformBufferSize = 64; // A 4x4 matrix of f32 takes 64 bytes
  const lightUniformBufferSize = 32; // 8 floats * 4 bytes

  const groundUBO = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const sphereUBO = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const lightUBO = device.createBuffer({
    size: lightUniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Setup light data
  let lightDirection = [0, 1, 1];
  const len = Math.hypot(lightDirection[0], lightDirection[1], lightDirection[2]);
  lightDirection = [lightDirection[0] / len, lightDirection[1] / len, lightDirection[2] / len];

  const lightColor = [0.3, 0.8, 1.0];

  const lightDataArray = new Float32Array([
    lightDirection[0], lightDirection[1], lightDirection[2], 0.0,
    lightColor[0], lightColor[1], lightColor[2], 0.0
  ]);
  device.queue.writeBuffer(lightUBO, 0, lightDataArray);

  return { groundUBO, sphereUBO, lightUBO };
}

function createBindGroups(
  device: GPUDevice,
  pipeline: GPURenderPipeline,
  groundUBO: GPUBuffer,
  sphereUBO: GPUBuffer,
  lightUBO: GPUBuffer
) {
  const groundBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: groundUBO } },
      { binding: 1, resource: { buffer: lightUBO } }
    ]
  });

  const sphereBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sphereUBO } },
      { binding: 1, resource: { buffer: lightUBO } }
    ]
  });

  return { groundBindGroup, sphereBindGroup };
}

function updateUniforms() {
  // Update MVP for ground
  const groundMVP = mat4.mul(perspective, view);
  mat4.mul(groundMVP, groundModelMatrix, groundMVP);
  device.queue.writeBuffer(groundUniformBuffer, 0, groundMVP as Float32Array);

  // Update MVP for sphere
  const sphereModelMatrix = mat4.translation(sphereCoordinates)
  const sphereMVP = mat4.mul(perspective, view);
  mat4.mul(sphereMVP, sphereModelMatrix, sphereMVP);
  device.queue.writeBuffer(sphereUniformBuffer, 0, sphereMVP as Float32Array);
}

function updatePhysicsSimulation() {
  world.gravity = { x: 0.0, y: -9.81, z: 0.0 }
  world.step();
  let position = sphereRigidBody.translation();
  sphereCoordinates = vec3.fromValues(position.x, position.y, position.z)
}

function render() {
  // Step the simulation forward.  
  updatePhysicsSimulation()
  updateUniforms();
  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [{
      view: msaaTexture.createView(),
      resolveTarget: context.getCurrentTexture().createView(),
      loadOp: 'clear',
      storeOp: 'store',
      clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1.0 },
    }],
    depthStencilAttachment: {
      view: depthStencil.createView(),
      depthLoadOp: 'clear',
      depthStoreOp: 'store',
      depthClearValue: 1.0,
    }
  });

  pass.setPipeline(pipeline);

  drawObject(pass, groundBuffer, groundBindGroup, ground.length / 8);
  drawObject(pass, sphereBuffer, sphereBindGroup, sphere.length / 8);

  pass.end();

  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
}

function drawObject(pass: GPURenderPassEncoder, vertexBuffer: GPUBuffer, bindGroup: GPUBindGroup, vertexCount: number) {
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(vertexCount, 1, 0, 0);
}
function initRapier() {
  let gravity = { x: 0.0, y: -9.81, z: 0.0 };
  world = new RAPIER.World(gravity);

  // Create the ground
  let groundColliderDesc = RAPIER.ColliderDesc.cuboid(gX, gY, gZ);
  world.createCollider(groundColliderDesc);

  // Create the sphere 
  let sphereRigidBodyDesc = RAPIER.RigidBodyDesc.dynamic()
    .setTranslation(sphereCoordinates[0], sphereCoordinates[1], sphereCoordinates[2]);
  sphereRigidBody = world.createRigidBody(sphereRigidBodyDesc);

  // Create a sphere collider attached to the rigidBody.
  let colliderDesc = RAPIER.ColliderDesc.ball(1).setRestitution(0.9);
  let sphereCollider = world.createCollider(colliderDesc, sphereRigidBody);
}
