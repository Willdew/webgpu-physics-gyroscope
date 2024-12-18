import "./style.css";
import vert from "./shaders/shader.wgsl?raw";
import { mat4, vec3, quat } from "wgpu-matrix";
import * as webgpuHelper from "./common/webgpuHelper.ts";
import RAPIER from "@dimforge/rapier3d-compat";

type viewType = "follow" | "static" | "staticFollow";
RAPIER.init().then(() => {
  // ------------------ DOM ELEMENTS ------------------
  const canvas = document.getElementById("canvas") as HTMLCanvasElement;
  const quaternionDisplay = document.getElementById(
    "quaternion-display",
  ) as HTMLDivElement;
  const resetButton = document.getElementById("resetBase");
  const viewButton = document.getElementById("view-mode");

  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  // ------------------ STATE ------------------
  // GPU-related
  let device: GPUDevice;
  let context: GPUCanvasContext;
  let format: GPUTextureFormat;
  let pipeline: GPURenderPipeline;
  let groundBuffer: GPUBuffer;
  let sphereBuffer: GPUBuffer;
  let groundUniformBuffer: GPUBuffer;
  let sphereUniformBuffer: GPUBuffer;
  let groundBindGroup: GPUBindGroup;
  let sphereBindGroup: GPUBindGroup;
  let depthStencil: GPUTexture;
  let msaaTexture: GPUTexture;

  // Physics-related
  let world: RAPIER.World;
  let sphereRigidBody: RAPIER.RigidBody;
  let groundRigidBody: RAPIER.RigidBody;
  let gravityY = -9.81;
  let currentRestitution = 0.5;
  let accelerometerFactor = 0.2;

  // Orientation sensor related
  let orientationSensor: RelativeOrientationSensor;
  let linearAccelerationSensor: LinearAccelerationSensor;
  let groundRotationQuat = quat.identity();
  let groundRotationMatrix = mat4.identity();
  let groundRotationEuler: { pitch: number; yaw: number; roll: number };
  let orientationSensorQuaternionRaw = [0, 0, 0, 0];
  let worldAccelerometer = [0, 0, 0];
  let baseOffset = 0;
  let resetBase = false;
  let mouse = false;
  let sphereCollider: any;

  // Scene
  let aspect = canvas.width / canvas.height;
  let sphereCoordinates = vec3.fromValues(0, 10, 0);
  let sphereRotation = quat.identity();
  const [gX, gY, gZ] = [40, 8, 40];
  const ground = webgpuHelper.createCuboidVertices(gX, gY, gZ);
  const sphere = webgpuHelper.createSphereVertices(32);
  let viewMode: viewType = "follow";
  let eye: number[];

  const gravitySlider = document.getElementById(
    "gravity-slider",
  ) as HTMLInputElement;
  const bouncinessSlider = document.getElementById(
    "bounciness-slider",
  ) as HTMLInputElement;
  const accelFactorSlider = document.getElementById(
    "accel-factor-slider",
  ) as HTMLInputElement;

  const gravityValueDisplay = document.getElementById(
    "gravity-value",
  ) as HTMLSpanElement;
  const bouncinessValueDisplay = document.getElementById(
    "bounciness-value",
  ) as HTMLSpanElement;
  const accelFactorValueDisplay = document.getElementById(
    "accel-factor-value",
  ) as HTMLSpanElement;

  gravityValueDisplay.textContent = gravitySlider.value;
  bouncinessValueDisplay.textContent = bouncinessSlider.value;
  accelFactorValueDisplay.textContent = accelFactorSlider.value;

  // ------------------ INITIALIZATION ------------------
  (async () => {
    await initializeWebGPU();
    initializeOrientationSensor();
    initializeAccelerometer();
    initializePipeline();
    initializeBuffers(ground, sphere);
    initializeUniformBuffers();
    initializeBindGroups();
    initializePhysics();
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

  function initializeBuffers(
    groundVertices: Float32Array,
    sphereVertices: Float32Array,
  ) {
    groundBuffer = createVertexBuffer(device, groundVertices);
    sphereBuffer = createVertexBuffer(device, sphereVertices);
  }

  function initializeUniformBuffers() {
    const uniformBufferSize = 64 * 3 + 16; // for a 4x4 matrix

    groundUniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    sphereUniformBuffer = device.createBuffer({
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  function initializeBindGroups() {
    groundBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: groundUniformBuffer } }],
    });

    sphereBindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: sphereUniformBuffer } }],
    });
  }

  function initializePhysics() {
    const gravity = { x: 0.0, y: -9.81, z: 0.0 };
    world = new RAPIER.World(gravity);

    // Ground
    const groundRigidBodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased();
    groundRigidBody = world.createRigidBody(groundRigidBodyDesc);

    const groundColliderDesc = RAPIER.ColliderDesc.cuboid(
      gX / 2,
      gY / 2,
      gZ / 2,
    );
    world.createCollider(groundColliderDesc, groundRigidBody);

    // Sphere
    const sphereRigidBodyDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(
        sphereCoordinates[0],
        sphereCoordinates[1],
        sphereCoordinates[2],
      )
      .lockTranslations() // prevent translations along along all axes.
      .lockRotations(); // prevent rotations along all axes.;
    sphereRigidBody = world.createRigidBody(sphereRigidBodyDesc);

    const sphereColliderDesc = RAPIER.ColliderDesc.ball(1).setRestitution(0.5);
    sphereCollider = world.createCollider(sphereColliderDesc, sphereRigidBody);
    // Schedule Unlocking of Translations and Rotations After 500ms
    setTimeout(() => {
      // Unlock Translations on all axes
      resetBase = true;
      sphereRigidBody.lockTranslations(false, true);

      // Unlock Rotations on all axes
      sphereRigidBody.lockRotations(false, true);
    }, 500);
    setInvisibleWalls(true);
  }

  let wallHeight = 50;
  let wallThickness = 0.1;
  let invisibleWalls = [];
  function setInvisibleWalls(state: boolean) {
    if (state) {
      let colliderBack = RAPIER.ColliderDesc.cuboid(
        gX / 2,
        wallHeight,
        wallThickness,
      ).setTranslation(0, wallHeight / 2, gZ / 2);
      invisibleWalls.push(world.createCollider(colliderBack));

      let colliderFront = RAPIER.ColliderDesc.cuboid(
        gX / 2,
        wallHeight,
        wallThickness,
      ).setTranslation(0, wallHeight / 2, -(gZ / 2));
      invisibleWalls.push(world.createCollider(colliderFront));

      let colliderLeft = RAPIER.ColliderDesc.cuboid(
        wallThickness,
        wallHeight,
        gZ / 2,
      ).setTranslation(gX / 2, wallHeight / 2, 0);
      invisibleWalls.push(world.createCollider(colliderLeft));

      let colliderRight = RAPIER.ColliderDesc.cuboid(
        wallThickness,
        wallHeight,
        gZ / 2,
      ).setTranslation(-(gX / 2), wallHeight / 2, 0);
      invisibleWalls.push(world.createCollider(colliderRight));
      let colliderTop = RAPIER.ColliderDesc.cuboid(
        gX / 2,
        2,
        gZ / 2,
      ).setTranslation(0, wallHeight / 2, 0);
      invisibleWalls.push(world.createCollider(colliderTop));
    }
  }

  function initializeOrientationSensor() {
    if (!("RelativeOrientationSensor" in window)) {
      console.error("AbsoluteOrientationSensor not supported.");
      if (quaternionDisplay) {
        quaternionDisplay.textContent =
          "AbsoluteOrientationSensor not supported by this browser.";
      }
      mouse = true;
      return;
    }

    const options: MotionSensorOptions = {
      frequency: 60,
      referenceFrame: "screen",
    };
    orientationSensor = new RelativeOrientationSensor(options);

    orientationSensor.addEventListener("reading", orientationSensorCallback);
    orientationSensor.addEventListener("error", handleSensorError);

    getSensorPermissions(orientationSensor);
  }
  function initializeAccelerometer() {
    // if (!("linearAccelerationSensor" in window)) {
    //   console.error("accelerometer not supported.");
    //   return;
    // }

    linearAccelerationSensor = new LinearAccelerationSensor({ frequency: 60 });

    linearAccelerationSensor.addEventListener("reading", accelerometerCallback);
    linearAccelerationSensor.addEventListener("error", handleSensorError);

    tryStartAccelerometer(linearAccelerationSensor);
  }

  // ------------------ RENDERING AND UPDATE ------------------

  function render() {
    updateUniforms();
    updatePhysics();

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: msaaTexture.createView(),
          resolveTarget: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0.0, g: 1, b: 1, a: 1.0 },
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
    drawObject(pass, groundBuffer, groundBindGroup, ground.length / 8);
    drawObject(pass, sphereBuffer, sphereBindGroup, sphere.length / 8);

    pass.end();
    device.queue.submit([commandEncoder.finish()]);

    requestAnimationFrame(render);
  }

  // ------------------ HELPER FUNCTIONS ------------------

  function createVertexBuffer(
    device: GPUDevice,
    data: Float32Array,
  ): GPUBuffer {
    const buffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  function updateUniforms() {
    const { perspective, view } = calculateView();

    // Ground MVP
    device.queue.writeBuffer(
      groundUniformBuffer,
      0,
      groundRotationMatrix as Float32Array,
    ); // Model
    device.queue.writeBuffer(groundUniformBuffer, 64, view as Float32Array); // View
    device.queue.writeBuffer(
      groundUniformBuffer,
      128,
      perspective as Float32Array,
    ); // Projection
    device.queue.writeBuffer(groundUniformBuffer, 192, new Float32Array(eye)); // Projection

    // Sphere MVP
    const sphereModelMatrix = mat4.mul(
      mat4.translation(sphereCoordinates),
      mat4.fromQuat(sphereRotation),
    );

    device.queue.writeBuffer(
      sphereUniformBuffer,
      0,
      sphereModelMatrix as Float32Array,
    ); // Model
    device.queue.writeBuffer(sphereUniformBuffer, 64, view as Float32Array); // View
    device.queue.writeBuffer(
      sphereUniformBuffer,
      128,
      perspective as Float32Array,
    ); // Projection
    device.queue.writeBuffer(sphereUniformBuffer, 192, new Float32Array(eye)); // Projection
  }

  function calculateView() {
    const fov = (90 * Math.PI) / 180;
    const near = 0.1;
    const far = 1000;
    const perspective = mat4.perspective(fov, aspect, near, far);
    let target: any;
    // const eye = [sphereCoordinates[0], 10, sphereCoordinates[2] + 15];
    switch (viewMode) {
      case "follow":
        eye = [
          sphereCoordinates[0],
          sphereCoordinates[1] + 10,
          sphereCoordinates[2] + 15,
        ];
        target = sphereCoordinates;
        break;

      case "static":
        eye = [0, 15, 25];
        target = [0, 0, 0];
        break;

      case "staticFollow":
        eye = [0, 15, 25];
        target = sphereCoordinates;
        break;

      default:
        eye = [0, 15, 25];
        target = [0, 0, 0];
        break;
    }

    const up = [0, 1, 0];
    const view = mat4.lookAt(eye, target, up);
    return { perspective, view };
  }

  function updatePhysics() {
    groundRigidBody.setNextKinematicRotation({
      x: groundRotationQuat[0],
      y: groundRotationQuat[1],
      z: groundRotationQuat[2],
      w: groundRotationQuat[3],
    });

    // Apply accelerometer factor (previously hard-coded to 0.2)
    groundRigidBody.setNextKinematicTranslation({
      x: worldAccelerometer[0] * accelerometerFactor,
      y: worldAccelerometer[2] * accelerometerFactor,
      z: worldAccelerometer[1] * accelerometerFactor,
    });

    if (!checkSphereBounds()) {
      sphereRigidBody.setTranslation({ x: 0, y: 10, z: 0 }, true);
    }

    world.step();

    const position = sphereRigidBody.translation();
    const rotation = sphereRigidBody.rotation();

    sphereCoordinates = vec3.fromValues(position.x, position.y, position.z);
    sphereRotation = quat.fromValues(
      rotation.x,
      rotation.y,
      rotation.z,
      rotation.w,
    );
  }

  function checkSphereBounds() {
    return sphereCoordinates[1] < 100 && sphereCoordinates[1] > -100;
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

  function orientationSensorCallback() {
    orientationSensorQuaternionRaw = orientationSensor.quaternion || [
      0, 0, 0, 0,
    ];
    groundRotationEuler = quatToEuler(orientationSensorQuaternionRaw);

    groundRotationQuat = quat.fromEuler(
      -groundRotationEuler.pitch + 0.5 * Math.PI + baseOffset,
      0,
      groundRotationEuler.roll,
      "xyz",
    );
    groundRotationMatrix = mat4.fromQuat(groundRotationQuat);

    if (resetBase) {
      baseOffset = groundRotationEuler.pitch - 0.5 * Math.PI;
      resetBase = false;
    }

    updateEulerDisplay();
  }

  function accelerometerCallback() {
    let x = linearAccelerationSensor.x || 0;
    let y = linearAccelerationSensor.y || 0;
    let z = linearAccelerationSensor.z || 0;

    let accRaw = vec3.fromValues(x, y, z);
    let orientationQuat = quat.fromValues(
      orientationSensorQuaternionRaw[0],
      orientationSensorQuaternionRaw[1],
      orientationSensorQuaternionRaw[2],
      orientationSensorQuaternionRaw[3],
    );

    worldAccelerometer = vec3.transformQuat(accRaw, orientationQuat);
  }

  function quatToEuler(inputQuat: number[]) {
    const [q0, q1, q2, q3] = inputQuat;
    const yaw = Math.atan2(
      2 * (q0 * q1 + q2 * q3),
      1 - 2 * (q1 * q1 + q2 * q2),
    );
    const roll = Math.asin(2 * (q0 * q2 - q3 * q1));
    const pitch = Math.atan2(
      2 * (q0 * q3 + q1 * q2),
      1 - 2 * (q2 * q2 + q3 * q3),
    );
    return { pitch, yaw, roll };
  }

  function updateEulerDisplay() {
    if (!quaternionDisplay || !groundRotationEuler) return;
    quaternionDisplay.textContent =
      `Euler:\n` +
      `Pitch: ${groundRotationEuler.pitch.toFixed(4)}\n` +
      `Yaw: ${groundRotationEuler.yaw.toFixed(4)}\n` +
      `Roll: ${groundRotationEuler.roll.toFixed(4)}\n` +
      `x: ${worldAccelerometer[0].toFixed(4)}\n` +
      `y: ${worldAccelerometer[1].toFixed(4)}\n` +
      `z: ${worldAccelerometer[2].toFixed(4)}\n`;
  }

  function handleSensorError(event: any) {
    mouse = true;
    if (event.error.name === "NotAllowedError") {
      console.error("Sensor access was denied by the user.");
    } else if (event.error.name === "NotReadableError") {
      console.error(
        "Cannot connect to the sensor. It may be in use by another application.",
      );
    } else {
      console.error("Sensor encountered an error:", event.error);
    }
  }

  function getSensorPermissions(sensor: RelativeOrientationSensor) {
    if (!navigator.permissions) {
      console.warn(
        "Permissions API not supported. Attempting to start sensor anyway.",
      );
      try {
        sensor.start();
        console.log("AbsoluteOrientationSensor started.");
      } catch (error) {
        console.error("Failed to start AbsoluteOrientationSensor:", error);
      }
      return;
    }

    Promise.all([
      navigator.permissions.query({ name: "accelerometer" as PermissionName }),
      navigator.permissions.query({ name: "magnetometer" as PermissionName }),
      navigator.permissions.query({ name: "gyroscope" as PermissionName }),
    ])
      .then((results) => {
        if (results.every((r) => r.state === "granted")) {
          tryStartSensor(sensor);
        } else if (results.some((r) => r.state === "prompt")) {
          showStartSensorButton(sensor);
        } else {
          console.log("No permissions to use AbsoluteOrientationSensor.");
        }
      })
      .catch((error) => {
        console.error("Error querying permissions:", error);
      });
  }

  function tryStartSensor(sensor: RelativeOrientationSensor) {
    try {
      sensor.start();
      console.log("AbsoluteOrientationSensor started.");
    } catch (error) {
      console.error("Failed to start AbsoluteOrientationSensor:", error);
    }
  }

  //Types are badly organized so that they cannot implement the same interface
  function tryStartAccelerometer(sensor: LinearAccelerationSensor) {
    try {
      sensor.start();
      console.log("accelerometer started.");
    } catch (error) {
      console.error("Failed to start accelerometer:", error);
    }
  }

  function showStartSensorButton(sensor: RelativeOrientationSensor) {
    const startButton = document.createElement("button");
    startButton.textContent = "Enable Orientation Sensor";
    Object.assign(startButton.style, {
      position: "absolute",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      padding: "10px 20px",
      fontSize: "16px",
      zIndex: "20",
    });
    document.body.appendChild(startButton);

    startButton.addEventListener("click", () => {
      try {
        sensor.start();
        console.log("AbsoluteOrientationSensor started.");
        document.body.removeChild(startButton);
      } catch (error) {
        console.error("Failed to start AbsoluteOrientationSensor:", error);
      }
    });
  }

  // ------------------ EVENT LISTENERS ------------------

  window.addEventListener("resize", () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    aspect = canvas.width / canvas.height;
    depthStencil = webgpuHelper.createDepthStencil(canvas, device);
    msaaTexture = webgpuHelper.createMSAATexture(canvas, device, format);
  });

  resetButton?.addEventListener("click", () => {
    resetBase = true;
  });

  viewButton?.addEventListener("click", () => {
    switch (viewMode) {
      case "follow":
        viewMode = "static";
        break;
      case "static":
        viewMode = "staticFollow";
        break;
      case "staticFollow":
        viewMode = "follow";
        break;
      default:
        break;
    }
  });

  let mouseCenterX = 0;
  let mouseCenterY = 0;

  function updateMouseCenter() {
    mouseCenterX = canvas.width / 2;
    mouseCenterY = canvas.height / 2;
  }

  updateMouseCenter();

  const rotationScale = 0.004;

  canvas.addEventListener("mousemove", (event) => {
    if (!mouse) return; // Only run this if mouse mode is active

    const offsetX = event.clientX - mouseCenterX;
    const offsetY = event.clientY - mouseCenterY;

    const angleX = offsetY * rotationScale;
    const angleZ = offsetX * rotationScale;

    const mouseQuat = quat.fromEuler(angleX, 0, -angleZ, "xyz");

    groundRotationQuat = mouseQuat;
    groundRotationMatrix = mat4.fromQuat(mouseQuat);
  });

  document.addEventListener("keydown", (event) => {
    if (event.code === "Space") {
      event.preventDefault(); // Prevent default spacebar scrolling behavior
      mouse = !mouse;
    }
  });

  gravitySlider.addEventListener("input", () => {
    gravityY = parseFloat(gravitySlider.value);
    gravityValueDisplay.textContent = gravitySlider.value;
    // Update the physics world gravity directly
    world.gravity = { x: 0, y: gravityY, z: 0 };
  });

  bouncinessSlider.addEventListener("input", () => {
    currentRestitution = parseFloat(bouncinessSlider.value);
    bouncinessValueDisplay.textContent = bouncinessSlider.value;
    // Update sphere's restitution if collider is accessible
    if (sphereCollider) {
      sphereCollider.setRestitution(currentRestitution);
    }
  });

  accelFactorSlider.addEventListener("input", () => {
    accelerometerFactor = parseFloat(accelFactorSlider.value);
    accelFactorValueDisplay.textContent = accelFactorSlider.value;
  });
});
