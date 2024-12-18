(function(){const A=document.createElement("link").relList;if(A&&A.supports&&A.supports("modulepreload"))return;for(const C of document.querySelectorAll('link[rel="modulepreload"]'))g(C);new MutationObserver(C=>{for(const S of C)if(S.type==="childList")for(const K of S.addedNodes)K.tagName==="LINK"&&K.rel==="modulepreload"&&g(K)}).observe(document,{childList:!0,subtree:!0});function I(C){const S={};return C.integrity&&(S.integrity=C.integrity),C.referrerPolicy&&(S.referrerPolicy=C.referrerPolicy),C.crossOrigin==="use-credentials"?S.credentials="include":C.crossOrigin==="anonymous"?S.credentials="omit":S.credentials="same-origin",S}function g(C){if(C.ep)return;C.ep=!0;const S=I(C);fetch(C.href,S)}})();const IC=`struct VertexIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>
};

@group(0) @binding(0) var<uniform> mvpMatrix : mat4x4<f32>;

struct LightData {
    direction: vec3<f32>,
    _pad1: f32,
    color: vec3<f32>,
    _pad2: f32,
};

@group(0) @binding(1) var<uniform> lightData : LightData;

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    output.position = mvpMatrix * input.position;
    output.normal = normalize(input.normal.xyz);
    return output;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let N = normalize(input.normal);
    let L = normalize(lightData.direction);
    let diffuse = max(dot(N, L), 0.0);

    // Simple greyish material
    let baseColor = vec3<f32>(N[0] * 0.8, N[1] * 0.8, 0.8);
    let shadedColor = baseColor * (diffuse * lightData.color);
    return vec4<f32>(shadedColor, 1.0);
}
Pitch: ${oA.pitch.toFixed(4)}
Yaw: ${oA.yaw.toFixed(4)}
Roll: ${oA.roll.toFixed(4)}
x: ${kA[0].toFixed(4)}
y: ${kA[1].toFixed(4)}
z: ${kA[2].toFixed(4)}
`)}function J(M){M.error.name==="NotAllowedError"?console.error("Sensor access was denied by the user."):M.error.name==="NotReadableError"?console.error("Cannot connect to the sensor. It may be in use by another application."):console.error("Sensor encountered an error:",M.error)}function q(M){if(!navigator.permissions){console.warn("Permissions API not supported. Attempting to start sensor anyway.");try{M.start(),console.log("AbsoluteOrientationSensor started.")}catch(a){console.error("Failed to start AbsoluteOrientationSensor:",a)}return}Promise.all([navigator.permissions.query({name:"accelerometer"}),navigator.permissions.query({name:"magnetometer"}),navigator.permissions.query({name:"gyroscope"})]).then(a=>{a.every(t=>t.state==="granted")?n(M):a.some(t=>t.state==="prompt")?U(M):console.log("No permissions to use AbsoluteOrientationSensor.")}).catch(a=>{console.error("Error querying permissions:",a)})}function n(M){try{M.start(),console.log("AbsoluteOrientationSensor started.")}catch(a){console.error("Failed to start AbsoluteOrientationSensor:",a)}}function h(M){try{M.start(),console.log("accelerometer started.")}catch(a){console.error("Failed to start accelerometer:",a)}}function U(M){const a=document.createElement("button");a.textContent="Enable Orientation Sensor",Object.assign(a.style,{position:"absolute",top:"50%",left:"50%",transform:"translate(-50%, -50%)",padding:"10px 20px",fontSize:"16px",zIndex:"20"}),document.body.appendChild(a),a.addEventListener("click",()=>{try{M.start(),console.log("AbsoluteOrientationSensor started."),document.body.removeChild(a)}catch(t){console.error("Failed to start AbsoluteOrientationSensor:",t)}})}window.addEventListener("resize",()=>{Q.width=window.innerWidth,Q.height=window.innerHeight,u=Q.width/Q.height,V=DC(Q,g),_=oC(Q,g,S)}),I==null||I.addEventListener("click",()=>{lA=!0})});