(function(){const A=document.createElement("link").relList;if(A&&A.supports&&A.supports("modulepreload"))return;for(const C of document.querySelectorAll('link[rel="modulepreload"]'))g(C);new MutationObserver(C=>{for(const S of C)if(S.type==="childList")for(const h of S.addedNodes)h.tagName==="LINK"&&h.rel==="modulepreload"&&g(h)}).observe(document,{childList:!0,subtree:!0});function I(C){const S={};return C.integrity&&(S.integrity=C.integrity),C.referrerPolicy&&(S.referrerPolicy=C.referrerPolicy),C.crossOrigin==="use-credentials"?S.credentials="include":C.crossOrigin==="anonymous"?S.credentials="omit":S.credentials="same-origin",S}function g(C){if(C.ep)return;C.ep=!0;const S=I(C);fetch(C.href,S)}})();const CQ=`struct VertexIn {
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
Pitch: ${I.toFixed(4)}
Yaw: ${g.toFixed(4)}
Roll: ${C.toFixed(4)}`}}const $A=document.getElementById("canvas"),_Q=document.getElementById("quaternion-display"),Ng=document.getElementById("resetBase");$A.width=window.innerWidth;$A.height=window.innerHeight;let XA,xB,_I,pg,WB,mB,ng,rg,zC,fB,jB,dg,Tg,VB,Og,XB,vC=0,Mg=!1,Zg=1,lI,RI=$C.identity(),uB=kI.identity(),PB={pitch:0,yaw:0,roll:0};console.log(PB);console.log(RI);let Jg=AB.fromValues(0,10,0);const[zB,vB,_B]=[20,.01,20],$B=yQ(zB,vB,_B),AQ=NQ(32);bB.init().then(()=>{$Q().then(()=>requestAnimationFrame(IQ))});async function $Q(){({device:XA,context:xB,format:_I}=await JQ($A)),Zg=$A.width/$A.height,dg=NC($A,XA),Tg=MC($A,XA,_I),WB=uC(XA,$B),mB=uC(XA,AQ),pg=mQ(XA,_I,CQ);const{groundUBO:B,sphereUBO:A,lightUBO:I}=fQ(XA);ng=B,rg=A,zC=I,{world:Og,sphereRigidBody:XB,groundCollider:VB}=VQ(bB,{sphereCoordinates:Jg,gX:zB,gY:vB,gZ:_B}),lI=new RelativeOrientationSensor({frequency:30,referenceFrame:"device"}),lI.addEventListener("reading",()=>{XQ(lI,vC,Mg,_Q,{setGroundRotationEuler:C=>PB=C,setGroundRotationQuat:C=>RI=C,setGroundRotationMatrix:C=>uB=C,setBaseOffset:C=>vC=C,setResetBase:C=>Mg=C})}),lI.addEventListener("error",C=>{C.error.name==="NotAllowedError"?console.error("Sensor access was denied by the user."):C.error.name==="NotReadableError"?console.error("Cannot connect to the sensor. It may be in use by another application."):console.error("Sensor encountered an error:",C.error)}),uQ(lI),{groundBindGroup:fB,sphereBindGroup:jB}=jQ(XA,pg,ng,rg,zC),gQ(),window.addEventListener("resize",async()=>{$A.width=window.innerWidth,$A.height=window.innerHeight,Zg=$A.width/$A.height,dg=NC($A,XA),Tg=MC($A,XA,_I)}),Ng==null||Ng.addEventListener("click",function(){Mg=!0})}function IQ(){AE(),gQ();const B=XA.createCommandEncoder(),A=B.beginRenderPass({colorAttachments:[{view:Tg.createView(),resolveTarget:xB.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:.3,g:.3,b:.3,a:1}}],depthStencilAttachment:{view:dg.createView(),depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1}});A.setPipeline(pg),PC(A,WB,fB,$B.length/8),PC(A,mB,jB,AQ.length/8),A.end(),XA.queue.submit([B.finish()]),requestAnimationFrame(IQ)}function AE(){Og.gravity={x:0,y:-9.81,z:0},Og.step();let B=XB.translation();VB.setRotation({w:RI[3],x:RI[0],y:RI[1],z:RI[2]}),Jg=AB.fromValues(B.x,B.y,B.z)}function IE(){const B=90*Math.PI/180,g=kI.perspective(B,Zg,.1,1e3),C=[0,10,20],S=Jg,h=[0,1,0],y=kI.lookAt(C,S,h);return{perspective:g,view:y}}function gQ(){const{perspective:B,view:A}=IE(),I=kI.mul(B,A);kI.mul(I,uB,I),XA.queue.writeBuffer(ng,0,I);const g=kI.translation(Jg),C=kI.mul(B,A);kI.mul(C,g,C),XA.queue.writeBuffer(rg,0,C)}