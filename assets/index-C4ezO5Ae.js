(function(){const A=document.createElement("link").relList;if(A&&A.supports&&A.supports("modulepreload"))return;for(const B of document.querySelectorAll('link[rel="modulepreload"]'))g(B);new MutationObserver(B=>{for(const S of B)if(S.type==="childList")for(const K of S.addedNodes)K.tagName==="LINK"&&K.rel==="modulepreload"&&g(K)}).observe(document,{childList:!0,subtree:!0});function I(B){const S={};return B.integrity&&(S.integrity=B.integrity),B.referrerPolicy&&(S.referrerPolicy=B.referrerPolicy),B.crossOrigin==="use-credentials"?S.credentials="include":B.crossOrigin==="anonymous"?S.credentials="omit":S.credentials="same-origin",S}function g(B){if(B.ep)return;B.ep=!0;const S=I(B);fetch(B.href,S)}})();const AC=`struct VertexIn {
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
X: ${C.toFixed(4)}
Y: ${k.toFixed(4)}
Z: ${i.toFixed(4)}
W: ${o.toFixed(4)}`}}window.addEventListener("resize",async()=>{E.width=window.innerWidth,E.height=window.innerHeight,u=E.width/E.height,rA=SI.perspective(tA,u,iA,P),j=DC(E,g),_=oC(E,g,S)}),I==null||I.addEventListener("click",function(){m=!0})});