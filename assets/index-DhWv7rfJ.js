(function(){const A=document.createElement("link").relList;if(A&&A.supports&&A.supports("modulepreload"))return;for(const C of document.querySelectorAll('link[rel="modulepreload"]'))g(C);new MutationObserver(C=>{for(const S of C)if(S.type==="childList")for(const K of S.addedNodes)K.tagName==="LINK"&&K.rel==="modulepreload"&&g(K)}).observe(document,{childList:!0,subtree:!0});function I(C){const S={};return C.integrity&&(S.integrity=C.integrity),C.referrerPolicy&&(S.referrerPolicy=C.referrerPolicy),C.crossOrigin==="use-credentials"?S.credentials="include":C.crossOrigin==="anonymous"?S.credentials="omit":S.credentials="same-origin",S}function g(C){if(C.ep)return;C.ep=!0;const S=I(C);fetch(C.href,S)}})();const $g=`struct VertexIn {
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
X: ${J.toFixed(4)}
Y: ${N.toFixed(4)}
Z: ${t.toFixed(4)}
W: ${E.toFixed(4)}`}}window.addEventListener("resize",async()=>{Q.width=window.innerWidth,Q.height=window.innerHeight,hA=Q.width/Q.height,u=SI.perspective(aA,hA,GA,HA),O=iC(Q,I),j=DC(Q,I,C)})});