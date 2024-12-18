import "./style.css";
import vert from "./shaders/shader.wgsl?raw";
import { mat4, vec3, quat } from "wgpu-matrix";
import * as webgpuHelper from "./common/webgpuHelper.ts";

// ------------------ DOM ELEMENTS ------------------
const canvas = document.getElementById("canvas") as HTMLCanvasElement;
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// ------------------ STATE ------------------
// GPU-related
let device: GPUDevice;
let context: GPUCanvasContext;
let format: GPUTextureFormat;
let pipeline: GPURenderPipeline;
let groundBuffer: GPUBuffer;
let cubeUniformBuffer: GPUBuffer;
let cubeBindGroup: GPUBindGroup;
let depthStencil: GPUTexture;
let msaaTexture: GPUTexture;

// Scene
let aspect = canvas.width / canvas.height;
const cube = webgpuHelper.createCuboidVertices(1, 1, 1);
let eye: number[];
let quinc = quat.identity();
let qrot = quat.identity();

// ------------------ INITIALIZATION ------------------
(async () => {
  await initializeWebGPU();
  initializePipeline();
  initializeBuffers(cube);
  initializeUniformBuffers();
  initializeBindGroups();
  initEventHandlers(canvas);
  updateUniforms(); // initial uniforms
  requestAnimationFrame(render);
})();

// ------------------ INITIALIZATION FUNCTIONS ------------------

async function initializeWebGPU() {
  const result = await webgpuHelper.configCanvas(canvas);
  device = result.device;
  context = result.context;
  format = result.format;
  depthStencil = webgpuHelper.createDepthStencil(canvas, device);
  msaaTexture = webgpuHelper.createMSAATexture(canvas, device, format);
}

function initializePipeline() {
  pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module: device.createShaderModule({ code: vert }),
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: 32,
          attributes: [
            { shaderLocation: 0, format: "float32x4", offset: 0 }, // position
            { shaderLocation: 1, format: "float32x4", offset: 16 }, // normal
          ],
        },
      ],
    },
    fragment: {
      module: device.createShaderModule({ code: vert }),
      entryPoint: "fs_main",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
    multisample: { count: 4 },
  });
}

function initializeBuffers(cube: Float32Array) {
  groundBuffer = createVertexBuffer(device, cube);
}

function initializeUniformBuffers() {
  const uniformBufferSize = 64 * 3 + 16; // 64 bytes for each matrix (Model, View, Projection)

  cubeUniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

function initializeBindGroups() {
  cubeBindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: cubeUniformBuffer } }],
  });
}

// ------------------ RENDERING AND UPDATE ------------------

function render() {
  updateUniforms();

  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: msaaTexture.createView(),
        resolveTarget: context.getCurrentTexture().createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0.3, g: 0.3, b: 0.3, a: 1.0 },
      },
    ],
    depthStencilAttachment: {
      view: depthStencil.createView(),
      depthLoadOp: "clear",
      depthStoreOp: "store",
      depthClearValue: 1.0,
    },
  });

  pass.setPipeline(pipeline);
  drawObject(pass, groundBuffer, cubeBindGroup, cube.length / 8);
  pass.end();
  device.queue.submit([commandEncoder.finish()]);

  requestAnimationFrame(render);
}

// Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
// if we are away from the center of the sphere.
function project_to_sphere(x: number, y: number) {
  var r = 2;
  var d = Math.sqrt(x * x + y * y);
  var t = r * Math.sqrt(2);
  var z;
  if (d < r)
    // Inside sphere
    z = Math.sqrt(r * r - d * d);
  else if (d < t)
    z = 0; // On hyperbola
  else z = (t * t) / d;
  return z;
}

function initEventHandlers(canvas: HTMLCanvasElement) {
  var dragging = false; // Dragging or not
  var lastX = -1,
    lastY = -1; // Last position of the mouse

  canvas.onmousedown = function (ev) {
    // Mouse is pressed
    const target = ev.target as HTMLElement;
    var x = ev.clientX,
      y = ev.clientY;
    // Start dragging if a mouse is in <canvas>
    var rect = target.getBoundingClientRect();
    if (rect.left <= x && x < rect.right && rect.top <= y && y < rect.bottom) {
      lastX = x;
      lastY = y;
      dragging = true;
    }
  };

  canvas.onmouseup = function () {
    quinc = quat.identity();
    dragging = false;
  }; // Mouse is released

  canvas.onmousemove = function (ev) {
    // Mouse is moved

    const target = ev.target as HTMLElement;
    var x = ev.clientX,
      y = ev.clientY;
    if (dragging) {
      var rect = target.getBoundingClientRect();
      var s_x = ((x - rect.left) / rect.width - 0.5) * 2;
      var s_y = (0.5 - (y - rect.top) / rect.height) * 2;
      var s_last_x = ((lastX - rect.left) / rect.width - 0.5) * 2;
      var s_last_y = (0.5 - (lastY - rect.top) / rect.height) * 2;
      var v1 = vec3.fromValues(s_x, s_y, project_to_sphere(s_x, s_y));
      var v2 = vec3.fromValues(
        s_last_x,
        s_last_y,
        project_to_sphere(s_last_x, s_last_y),
      );
      quinc = quat.rotationTo(vec3.normalize(v1), vec3.normalize(v2));
      qrot = quat.mul(qrot, quinc);
    }
    (lastX = x), (lastY = y);
  };
} // ------------------ HELPER FUNCTIONS ------------------

function createVertexBuffer(device: GPUDevice, data: Float32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function updateUniforms() {
  const { perspective, view } = calculateView();

  // Model matrix
  const modelMatrix = mat4.identity();

  // Write matrices to the uniform buffer
  device.queue.writeBuffer(cubeUniformBuffer, 0, modelMatrix as Float32Array); // Model
  device.queue.writeBuffer(cubeUniformBuffer, 64, view as Float32Array); // View
  device.queue.writeBuffer(cubeUniformBuffer, 128, perspective as Float32Array); // Projection
  device.queue.writeBuffer(cubeUniformBuffer, 192, new Float32Array(eye)); // Projection
}

function calculateView() {
  const fov = (90 * Math.PI) / 180;
  const near = 0.1;
  const far = 1000;
  const perspective = mat4.perspective(fov, aspect, near, far);
  let target: any;
  let eyeStart = new Float32Array([1, 1, 2]);
  eye = vec3.transformQuat(eyeStart, qrot);
  target = [0, 0, 0];

  const up = [0, 1, 0];
  const view = mat4.lookAt(eye, target, up);
  return { perspective, view };
}

function drawObject(
  pass: GPURenderPassEncoder,
  vertexBuffer: GPUBuffer,
  bindGroup: GPUBindGroup,
  vertexCount: number,
) {
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vertexBuffer);
  pass.draw(vertexCount, 1, 0, 0);
}

// ------------------ SENSOR AND UI ------------------

// ------------------ EVENT LISTENERS ------------------

window.addEventListener("resize", () => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  aspect = canvas.width / canvas.height;
  depthStencil = webgpuHelper.createDepthStencil(canvas, device);
  msaaTexture = webgpuHelper.createMSAATexture(canvas, device, format);
});
