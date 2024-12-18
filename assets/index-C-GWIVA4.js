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
Pitch: ${JA.pitch.toFixed(4)}
Yaw: ${JA.yaw.toFixed(4)}
Roll: ${JA.roll.toFixed(4)}
x: ${wA[0].toFixed(4)}
y: ${wA[1].toFixed(4)}
z: ${wA[2].toFixed(4)}
`)}function L(T){T.error.name==="NotAllowedError"?console.error("Sensor access was denied by the user."):T.error.name==="NotReadableError"?console.error("Cannot connect to the sensor. It may be in use by another application."):console.error("Sensor encountered an error:",T.error)}function n(T){if(!navigator.permissions){console.warn("Permissions API not supported. Attempting to start sensor anyway.");try{T.start(),console.log("AbsoluteOrientationSensor started.")}catch(O){console.error("Failed to start AbsoluteOrientationSensor:",O)}return}Promise.all([navigator.permissions.query({name:"accelerometer"}),navigator.permissions.query({name:"magnetometer"}),navigator.permissions.query({name:"gyroscope"})]).then(O=>{O.every(x=>x.state==="granted")?r(T):O.some(x=>x.state==="prompt")?z(T):console.log("No permissions to use AbsoluteOrientationSensor.")}).catch(O=>{console.error("Error querying permissions:",O)})}function r(T){try{T.start(),console.log("AbsoluteOrientationSensor started.")}catch(O){console.error("Failed to start AbsoluteOrientationSensor:",O)}}function m(T){try{T.start(),console.log("accelerometer started.")}catch(O){console.error("Failed to start accelerometer:",O)}}function z(T){const O=document.createElement("button");O.textContent="Enable Orientation Sensor",Object.assign(O.style,{position:"absolute",top:"50%",left:"50%",transform:"translate(-50%, -50%)",padding:"10px 20px",fontSize:"16px",zIndex:"20"}),document.body.appendChild(O),O.addEventListener("click",()=>{try{T.start(),console.log("AbsoluteOrientationSensor started."),document.body.removeChild(O)}catch(x){console.error("Failed to start AbsoluteOrientationSensor:",x)}})}window.addEventListener("resize",()=>{Q.width=window.innerWidth,Q.height=window.innerHeight,DA=Q.width/Q.height,gA=DC(Q,C),QA=oC(Q,C,K)}),I==null||I.addEventListener("click",()=>{_=!0}),g==null||g.addEventListener("click",()=>{switch(pA){case"follow":pA="static";break;case"static":pA="staticFollow";break;case"staticFollow":pA="follow";break}})});