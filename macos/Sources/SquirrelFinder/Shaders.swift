/// Metal source, compiled at launch (keeps `swift build` free of metal-toolchain steps).
let shaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float2 scale;
    float2 offset;
    float2 worldSize;
    float  time;
    float  lightCount;
    float4 cam;       // sin(pitch), cos(pitch), floor-centre plane y, clip height (game units)
    float4 cam2;      // room units -> game units, floor y (room units), depth gain, plane centre x
    float4 ambient;   // moonlight rgb
};

struct Light {
    float4 a;         // x, z (game floor coords), radius, intensity
    float4 color;     // rgb, height above the floor (game units)
};

struct Sprite {
    float2 center;
    float2 size;
    float4 tint;
    float  rotation;
    float  kind;
    float  param;
    float  seed;
    float4 uv;        // sub-rect of the texture: x, y, w, h
    float4 depthInfo; // depth at feet, feet plane y, depth slope per plane y, unused
    float4 gpos;      // floor position (game coords) used for lighting
};

struct VOut {
    float4 pos [[position]];
    float2 uv;
    float2 world;
    float4 tint;
    float  kind;
    float  param;
    float  seed;
    float4 rect;
    float2 gpos;
};

vertex VOut sprite_vs(uint vid [[vertex_id]], uint iid [[instance_id]],
                      constant Uniforms& u [[buffer(0)]],
                      constant Sprite* sp [[buffer(1)]]) {
    Sprite s = sp[iid];
    float2 q = float2((vid & 1) ? 1.0 : -1.0, (vid & 2) ? 1.0 : -1.0);
    float2 local = q * 0.5 * s.size;
    float c = cos(s.rotation), sn = sin(s.rotation);
    float2 r = float2(local.x * c - local.y * sn, local.x * sn + local.y * c);
    float2 w = s.center + r;
    VOut o;
    float z = s.depthInfo.x + s.depthInfo.z * (w.y - s.depthInfo.y);
    o.pos = float4(w * u.scale + u.offset, clamp(z, 0.0, 1.0), 1.0);
    // negative size.x mirrors the sprite, so derive uv from the unflipped corner
    o.uv = float2(q.x * sign(s.size.x), q.y) * 0.5 + 0.5;
    o.rect = s.uv;
    o.world = w;
    o.tint = s.tint;
    o.kind = s.kind;
    o.param = s.param;
    o.seed = s.seed;
    o.gpos = s.gpos.xy;
    return o;
}

float hash21(float2 p) {
    p = fract(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(float2 p) {
    float2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + float2(1, 0)), f.x),
               mix(hash21(i + float2(0, 1)), hash21(i + float2(1, 1)), f.x), f.y);
}

float fbm(float2 p) {
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p *= 2.03; a *= 0.5; }
    return s;
}

// Night lighting: cold moonlit ambient plus coloured pools (wizard lantern, ghosts, penguin beacon, torches).
// P is in game units: x, height above the floor, z.
float3 lighting(float3 P, constant Light* L, int n, float3 ambient) {
    float3 c = ambient;
    for (int i = 0; i < n; i++) {
        float3 lp = float3(L[i].a.x, L[i].color.w, L[i].a.y);
        float t = saturate(1.0 - length(P - lp) / L[i].a.z);
        c += L[i].color.rgb * (t * t * (3.0 - 2.0 * t)) * L[i].a.w;
    }
    return c;
}

fragment float4 sprite_fs(VOut in [[stage_in]],
                          texture2d<float> tex [[texture(0)]],
                          constant Uniforms& u [[buffer(0)]],
                          constant Light* L [[buffer(2)]]) {
    constexpr sampler nearS(address::clamp_to_edge, filter::nearest);
    constexpr sampler linS(address::clamp_to_edge, filter::linear);
    constexpr sampler repS(address::repeat, filter::nearest);
    int k = int(in.kind + 0.5);
    float2 p = in.uv * 2.0 - 1.0;
    int n = int(u.lightCount);

    if (k == 0 || k == 6) {                         // textured sprite (sheet frame); 6 = emissive (unlit)
        float2 suv = in.rect.xy + in.uv * in.rect.zw;
        float4 c = in.seed > 0.5 ? tex.sample(linS, suv) : tex.sample(nearS, suv);
        float3 rgb = c.rgb * in.tint.rgb;
        if (k == 0) rgb *= min(lighting(float3(in.gpos.x, 45.0, in.gpos.y), L, n, u.ambient.rgb), float3(1.25));
        rgb = mix(rgb, float3(0.9, 0.95, 1.0), clamp(in.param, 0.0, 1.0) * 0.55);
        if (c.a * in.tint.a < 0.01) discard_fragment();
        return float4(rgb, c.a * in.tint.a);
    }
    if (k == 1) {                                   // soft contact shadow
        float d = length(p);
        float a = 1.0 - smoothstep(0.3, 1.0, d);
        return float4(in.tint.rgb, a * in.tint.a);
    }
    if (k == 2) {                                   // soft particle
        float d = length(p);
        float a = 1.0 - smoothstep(0.0, 1.0, d);
        return float4(in.tint.rgb, a * in.tint.a);
    }
    if (k == 3) {                                   // ring
        float d = length(p);
        float ring = 1.0 - smoothstep(0.0, 0.10, abs(d - 0.86));
        float fill = (1.0 - smoothstep(0.0, 1.0, d)) * 0.22;
        float pulse = 0.65 + 0.35 * sin(u.time * 16.0);
        return float4(in.tint.rgb, (ring * 0.95 + fill) * pulse * in.tint.a);
    }
    if (k == 4) {                                   // night meadow (flat graveyard level)
        float2 wp = in.world;
        float3 g = tex.sample(repS, wp / 576.0).rgb * 1.5;
        g = mix(g, float3(dot(g, float3(0.3, 0.5, 0.2))), 0.25) * float3(0.85, 1.0, 1.05);
        float3 lit = lighting(float3(wp.x, 0.0, wp.y), L, n, u.ambient.rgb);
        float moon = fbm(wp * 0.0028 + float2(u.time * 0.012, u.time * 0.007));
        lit *= 0.85 + 0.35 * smoothstep(0.35, 0.8, moon);
        float3 col = g * lit;
        float fog = fbm(wp * 0.0042 + float2(u.time * 0.025, -u.time * 0.015));
        col += float3(0.10, 0.14, 0.24) * smoothstep(0.42, 0.85, fog) * 0.30;
        return float4(col, 1.0);
    }
    // vignette: deep night corners
    float r = length(p);
    float v = smoothstep(0.40, 1.30, r);
    return float4(0.005, 0.01, 0.04, v * 0.82);
}

// ---- voxel rooms

struct RVert {
    packed_float3 pos;
    packed_float3 normal;
    packed_float3 color;
    float emissive;
};

struct ROut {
    float4 pos [[position]];
    float3 world;     // game units: x, height above the floor, z
    float3 normal;
    float3 color;
    float emissive;
};

vertex ROut room_vs(uint vid [[vertex_id]],
                    constant Uniforms& u [[buffer(0)]],
                    constant RVert* v [[buffer(1)]]) {
    RVert a = v[vid];
    float S = u.cam2.x;
    float gx = a.pos.x * S, gz = a.pos.z * S, h = (a.pos.y - u.cam2.y) * S;
    float2 plane = float2(u.cam2.w + gx, u.cam.z + gz * u.cam.x - h * u.cam.y);
    ROut o;
    float z = 0.55 - u.cam2.z * (h * u.cam.x + gz * u.cam.y);
    o.pos = float4(plane * u.scale + u.offset, z, 1.0);
    o.world = float3(gx, h, gz);
    o.normal = float3(a.normal);
    o.color = float3(a.color);
    o.emissive = a.emissive;
    return o;
}

fragment float4 room_fs(ROut in [[stage_in]],
                        constant Uniforms& u [[buffer(0)]],
                        constant Light* L [[buffer(2)]]) {
    if (in.world.y > u.cam.w) discard_fragment();            // cut-away: nothing above the clip height
    float3 N = normalize(in.normal);
    float3 toCam = float3(0.0, u.cam.x, u.cam.y);            // direction to the camera (y up, z front)
    int n = int(u.lightCount);
    // Faces turned away from the camera are only seen through the cut: paint them as dark stone.
    if (dot(N, toCam) < 0.0) {
        float3 capLit = lighting(float3(in.world.x, 30.0, in.world.z), L, n, u.ambient.rgb);
        return float4(float3(0.12, 0.11, 0.13) * min(capLit, float3(1.0)), 1.0);
    }
    float3 lit = u.ambient.rgb * (0.75 + 0.25 * max(N.y, 0.0));
    for (int i = 0; i < n; i++) {
        float3 lp = float3(L[i].a.x, L[i].color.w, L[i].a.y);
        float3 d = lp - in.world;
        float dist = length(d);
        float t = saturate(1.0 - dist / L[i].a.z);
        float ndl = saturate(dot(N, d / max(dist, 1e-3)) * 0.65 + 0.35);
        lit += L[i].color.rgb * (t * t * (3.0 - 2.0 * t)) * L[i].a.w * ndl;
    }
    float3 rgb = in.color * min(lit, float3(1.3));
    rgb = mix(rgb, in.color * 1.6, in.emissive);              // torch flames, lava
    // low fog hugging the floor
    float fog = fbm(in.world.xz * 0.0045 + float2(u.time * 0.02, -u.time * 0.012));
    rgb += float3(0.05, 0.07, 0.12) * smoothstep(0.4, 0.85, fog) * (1.0 - smoothstep(0.0, 120.0, in.world.y)) * 0.5;
    return float4(rgb, 1.0);
}
"""
