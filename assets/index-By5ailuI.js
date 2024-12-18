(function(){const A=document.createElement("link").relList;if(A&&A.supports&&A.supports("modulepreload"))return;for(const C of document.querySelectorAll('link[rel="modulepreload"]'))g(C);new MutationObserver(C=>{for(const w of C)if(w.type==="childList")for(const K of w.addedNodes)K.tagName==="LINK"&&K.rel==="modulepreload"&&g(K)}).observe(document,{childList:!0,subtree:!0});function I(C){const w={};return C.integrity&&(w.integrity=C.integrity),C.referrerPolicy&&(w.referrerPolicy=C.referrerPolicy),C.crossOrigin==="use-credentials"?w.credentials="include":C.crossOrigin==="anonymous"?w.credentials="omit":w.credentials="same-origin",w}function g(C){if(C.ep)return;C.ep=!0;const w=I(C);fetch(C.href,w)}})();const IC=`struct Matrices {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    cameraPosition: vec3<f32>
};

struct VertexIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) vNormal: vec4<f32>,
    @location(1) vPos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> matrices : Matrices;

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    let mvMatrix = matrices.view * matrices.model;
    output.position = matrices.projection * mvMatrix * input.position;
    output.vNormal = matrices.model * input.normal;
    output.vPos = matrices.model * input.position;
    return output;
}

@fragment
fn fs_main(fragData: VertexOut) -> @location(0) vec4<f32> {
    let specularStrength = 0.5;
    let specularShininess = 19.0;
    let lightPosition = vec3(5.0, 20.0, 2.0);
    let diffuseLightStrength = 0.8;
    let ambientLightStrength = 0.8;
    let ambientLightColor = vec4(0.0, 0.7, 0.7, 1.0);

  //Ambient
    let ambientFinal = ambientLightColor * ambientLightStrength;

  //Diffuse
    let vNormal = normalize(fragData.vNormal.xyz);
    let vPosition = fragData.vPos.xyz;
    let vCameraPosition = matrices.cameraPosition;
    let lightDir = normalize(lightPosition - vPosition);
    let lightMagnitude = dot(vNormal, lightDir);
    let diffuseLightFinal: f32 = diffuseLightStrength * max(lightMagnitude, 0.0);

  //Specular
    let viewDir = normalize(vCameraPosition - vPosition);
    let reflectDir = reflect(-lightDir, vNormal);
    let spec = pow(max(dot(viewDir, reflectDir), 0.0), specularShininess);
    let specularFinal = specularStrength * spec;
    let lightFinal = specularFinal + diffuseLightFinal;
    return vec4(0.9, 0.5, 0.5, 1.0) * lightFinal + ambientFinal;
}
Pitch: ${hA.pitch.toFixed(4)}
Yaw: ${hA.yaw.toFixed(4)}
Roll: ${hA.roll.toFixed(4)}
x: ${P[0].toFixed(4)}
y: ${P[1].toFixed(4)}
z: ${P[2].toFixed(4)}
`)}function lA(Z){XA=!0,Z.error.name==="NotAllowedError"?console.error("Sensor access was denied by the user."):Z.error.name==="NotReadableError"?console.error("Cannot connect to the sensor. It may be in use by another application."):console.error("Sensor encountered an error:",Z.error)}function TA(Z){if(!navigator.permissions){console.warn("Permissions API not supported. Attempting to start sensor anyway.");try{Z.start(),console.log("AbsoluteOrientationSensor started.")}catch(b){console.error("Failed to start AbsoluteOrientationSensor:",b)}return}Promise.all([navigator.permissions.query({name:"accelerometer"}),navigator.permissions.query({name:"magnetometer"}),navigator.permissions.query({name:"gyroscope"})]).then(b=>{b.every(AA=>AA.state==="granted")?WA(Z):b.some(AA=>AA.state==="prompt")?fA(Z):console.log("No permissions to use AbsoluteOrientationSensor.")}).catch(b=>{console.error("Error querying permissions:",b)})}function WA(Z){try{Z.start(),console.log("AbsoluteOrientationSensor started.")}catch(b){console.error("Failed to start AbsoluteOrientationSensor:",b)}}function mA(Z){try{Z.start(),console.log("accelerometer started.")}catch(b){console.error("Failed to start accelerometer:",b)}}function fA(Z){const b=document.createElement("button");b.textContent="Enable Orientation Sensor",Object.assign(b.style,{position:"absolute",top:"50%",left:"50%",transform:"translate(-50%, -50%)",padding:"10px 20px",fontSize:"16px",zIndex:"20"}),document.body.appendChild(b),b.addEventListener("click",()=>{try{Z.start(),console.log("AbsoluteOrientationSensor started."),document.body.removeChild(b)}catch(AA){console.error("Failed to start AbsoluteOrientationSensor:",AA)}})}window.addEventListener("resize",()=>{Q.width=window.innerWidth,Q.height=window.innerHeight,$A=Q.width/Q.height,X=DC(Q,C),CA=oC(Q,C,K)}),I==null||I.addEventListener("click",()=>{_=!0}),g==null||g.addEventListener("click",()=>{switch(xA){case"follow":xA="static";break;case"static":xA="staticFollow";break;case"staticFollow":xA="follow";break}});let OA=0,ZA=0;function PA(){OA=Q.width/2,ZA=Q.height/2}PA();const jA=.004;Q.addEventListener("mousemove",Z=>{if(!XA)return;const b=Z.clientX-OA,wA=(Z.clientY-ZA)*jA,GA=b*jA,CI=RI.fromEuler(wA,0,-GA,"xyz");MA=CI,qA=UI.fromQuat(CI)}),document.addEventListener("keydown",Z=>{Z.code==="Space"&&(Z.preventDefault(),XA=!XA)}),S.addEventListener("input",()=>{oA=parseFloat(S.value),G.textContent=S.value,v.gravity={x:0,y:oA,z:0}}),M.addEventListener("input",()=>{j=parseFloat(M.value),y.textContent=M.value,dA&&dA.setRestitution(j)}),D.addEventListener("input",()=>{sA=parseFloat(D.value),a.textContent=D.value})});