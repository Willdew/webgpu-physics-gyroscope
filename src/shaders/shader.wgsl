struct Matrices {
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
