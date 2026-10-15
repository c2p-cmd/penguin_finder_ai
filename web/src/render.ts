/**
 * Three.js (WebGPURenderer + TSL) renderer.
 *
 * Scene layout, all in "game units" (1 unit = 1 plane pixel of the 1280x800 design plane):
 *   - the voxel room mesh (rooms) or a tiled ground quad (graveyard), lit by up to 32 point-light pools
 *   - ONE instanced billboard mesh holding every sprite/shadow/crack, written back-to-front so
 *     overlapping sprites keep the painter's order, depth-tested against the room
 *   - an overlay instanced mesh for particles and a vignette quad (no depth test)
 * The orthographic camera is tilted 58 degrees for rooms and looks straight down for the graveyard.
 */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { Game, STAGE, WORLD, type LightRec, type TexName } from './game';
import { ROOM_VERTEX_FLOATS, type RoomMesh } from './roomMesh';

// TSL's TypeScript types are very strict for node arithmetic; the graph is validated when it compiles.
const {
  Fn, Loop, attribute, clamp, dot, float, floor, length, max, min, mix, normalWorld, positionWorld,
  select, smoothstep, texture, uniform, uniformArray, uv, vec2, vec3, vec4, vertexColor,
} = TSL as any;

const MAX_SPRITES = 512;
const MAX_PARTICLES = 512;
const LIGHTS = 32;

/** Where each source image lives inside the packed atlas (pixels). */
const ATLAS_W = 2048;
const ATLAS_H = 3072;
const ATLAS_LAYOUT: Record<TexName | 'ground', { file: string; x: number; y: number }> = {
  penguin: { file: 'penguin.png', x: 0, y: 0 }, // 1024 x 2048
  wizardWalk: { file: 'wizard_walk.png', x: 1024, y: 0 }, // 768 x 1024
  ghostWalk: { file: 'ghost_walk.png', x: 1024, y: 1024 }, // 768 x 1024
  wizardIdle: { file: 'wizard_idle.png', x: 0, y: 2048 }, // 512 x 1024
  props: { file: 'props.png', x: 512, y: 2048 }, // 256 x 75
  cracks: { file: 'cracks.png', x: 768, y: 2048 }, // 200 x 200
  ground: { file: 'ground.png', x: 0, y: 0 }, // separate texture (repeats)
};

async function loadBitmap(url: string): Promise<ImageBitmap> {
  const blob = await (await fetch(url)).blob();
  return createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
}

export class Renderer {
  private renderer!: THREE.WebGPURenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.OrthographicCamera(-640, 640, 400, -400, 0.1, 8000);
  private room = new THREE.Mesh();
  private ground = new THREE.Mesh();
  private sprites!: THREE.InstancedMesh;
  private overlay!: THREE.InstancedMesh;
  private vignette = new THREE.Mesh();

  private atlasRect = new Map<TexName, [number, number, number, number]>(); // normalised
  private roomGeo = new WeakMap<RoomMesh, THREE.BufferGeometry>();
  private shownStage = -1;
  private flatMode = true;

  // uniforms
  private lightA = uniformArray(Array.from({ length: LIGHTS }, () => new THREE.Vector4(0, 0, 1, 0)), 'vec4');
  private lightC = uniformArray(Array.from({ length: LIGHTS }, () => new THREE.Vector4(0, 0, 0, 0)), 'vec4');
  private ambient = uniform(new THREE.Color(0.075, 0.105, 0.2));

  // per-instance attributes
  private aRect!: THREE.InstancedBufferAttribute;
  private aTint!: THREE.InstancedBufferAttribute;
  private aInfo!: THREE.InstancedBufferAttribute;
  private aGpos!: THREE.InstancedBufferAttribute;
  private oTint!: THREE.InstancedBufferAttribute;

  private last = performance.now();
  private startTime = performance.now();
  private fps = 0;
  private fpsFrames = 0;
  private fpsT = performance.now();

  constructor(private canvas: HTMLCanvasElement, private game: Game, private assetBase: string) {}

  get framesPerSecond() { return this.fps; }
  get backend(): string {
    const r = this.renderer as any;
    return r?.backend?.isWebGPUBackend ? 'WebGPU' : r?.backend?.isWebGLBackend ? 'WebGL2' : '?';
  }

  async init() {
    const renderer = new THREE.WebGPURenderer({ canvas: this.canvas, antialias: false, alpha: false });
    renderer.setClearColor(new THREE.Color(0.004, 0.008, 0.02), 1);
    await renderer.init();
    this.renderer = renderer;

    await this.buildTextures();
    this.buildMaterialsAndMeshes();
    this.onResize();
    new ResizeObserver(() => this.onResize()).observe(this.canvas);
  }

  // MARK: textures
  private atlasTex!: THREE.Texture;
  private groundTex!: THREE.Texture;

  private async buildTextures() {
    const names = Object.keys(ATLAS_LAYOUT) as (TexName | 'ground')[];
    const bitmaps = await Promise.all(names.map((n) => loadBitmap(this.assetBase + ATLAS_LAYOUT[n].file)));
    const canvas = document.createElement('canvas');
    canvas.width = ATLAS_W;
    canvas.height = ATLAS_H;
    const ctx = canvas.getContext('2d')!;
    ctx.imageSmoothingEnabled = false;
    names.forEach((n, i) => {
      const bmp = bitmaps[i];
      if (n === 'ground') {
        const gt = new THREE.Texture(bmp as unknown as HTMLImageElement);
        gt.colorSpace = THREE.SRGBColorSpace;
        gt.wrapS = gt.wrapT = THREE.RepeatWrapping;
        gt.magFilter = gt.minFilter = THREE.NearestFilter;
        gt.generateMipmaps = false;
        gt.needsUpdate = true;
        this.groundTex = gt;
        return;
      }
      const { x, y } = ATLAS_LAYOUT[n];
      ctx.drawImage(bmp, x, y);
      this.atlasRect.set(n, [x / ATLAS_W, y / ATLAS_H, bmp.width / ATLAS_W, bmp.height / ATLAS_H]);
    });
    const atlas = new THREE.CanvasTexture(canvas);
    atlas.colorSpace = THREE.SRGBColorSpace;
    atlas.flipY = false;
    atlas.generateMipmaps = false;
    atlas.magFilter = atlas.minFilter = THREE.LinearFilter; // pixel snapping happens in the shader
    atlas.needsUpdate = true;
    this.atlasTex = atlas;
  }

  // MARK: materials + meshes
  private buildMaterialsAndMeshes() {
    const lightA = this.lightA as any;
    const lightC = this.lightC as any;
    const ambient = this.ambient as any;
    const smooth = (t: any) => t.mul(t).mul(float(3).sub(t.mul(2)));

    /** moonlit ambient + coloured pools; P is in game units (x, height above the floor, z). */
    const lighting = Fn(([P]: any[]) => {
      const c = vec3(ambient).toVar();
      Loop(LIGHTS, ({ i }: any) => {
        const A = lightA.element(i);
        const C = lightC.element(i);
        const lp = vec3(A.x, C.w, A.y);
        const t = clamp(float(1).sub(length(P.sub(lp)).div(A.z)), 0, 1);
        c.addAssign(C.xyz.mul(smooth(t)).mul(A.w));
      });
      return c;
    });

    /** same pools, with wrap lighting against the surface normal (voxel walls and floor). */
    const lightingN = Fn(([P, N]: any[]) => {
      const c = vec3(ambient).mul(float(0.75).add(max(N.y, 0).mul(0.25))).toVar();
      Loop(LIGHTS, ({ i }: any) => {
        const A = lightA.element(i);
        const C = lightC.element(i);
        const lp = vec3(A.x, C.w, A.y);
        const d = lp.sub(P);
        const dist = length(d);
        const t = clamp(float(1).sub(dist.div(A.z)), 0, 1);
        const ndl = clamp(dot(N, d.div(max(dist, 0.001))).mul(0.65).add(0.35), 0, 1);
        c.addAssign(C.xyz.mul(smooth(t)).mul(A.w).mul(ndl));
      });
      return c;
    });

    // ---- sprites (one instanced mesh, painter-ordered)
    const geo = new THREE.PlaneGeometry(1, 1);
    this.aRect = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SPRITES * 4), 4);
    this.aTint = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SPRITES * 4), 4);
    this.aInfo = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SPRITES * 4), 4);
    this.aGpos = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SPRITES * 4), 4);
    for (const a of [this.aRect, this.aTint, this.aInfo, this.aGpos]) a.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aRect', this.aRect);
    geo.setAttribute('aTint', this.aTint);
    geo.setAttribute('aInfo', this.aInfo);
    geo.setAttribute('aGpos', this.aGpos);

    const spriteMat = new THREE.MeshBasicNodeMaterial();
    {
      const aRect = attribute('aRect', 'vec4');
      const aTint = attribute('aTint', 'vec4');
      const aInfo = attribute('aInfo', 'vec4'); // kind (1 = shadow), flash, smooth, lit
      const aG = attribute('aGpos', 'vec4'); // world x, world z
      const local = uv();
      const fromTop = vec2(local.x, float(1).sub(local.y));
      const suv = aRect.xy.add(fromTop.mul(aRect.zw));
      const atlasSize = vec2(ATLAS_W, ATLAS_H);
      const snapped = floor(suv.mul(atlasSize)).add(0.5).div(atlasSize); // nearest-neighbour for pixel art
      const sampleUv = select(aInfo.z.greaterThan(0.5), suv, snapped);
      const col = texture(this.atlasTex).sample(sampleUv);
      const lit = min(lighting(vec3(aG.x, 45, aG.y)), vec3(1.25, 1.25, 1.25));
      const spriteRgb = col.rgb.mul(aTint.rgb).mul(select(aInfo.w.greaterThan(0.5), lit, vec3(1, 1, 1)));
      const d = length(local.mul(2).sub(1));
      const shadowA = float(1).sub(smoothstep(0.3, 1.0, d));
      const isShadow = aInfo.x.greaterThan(0.5);
      const rgb = select(isShadow, aTint.rgb, spriteRgb);
      const flashed = mix(rgb, vec3(0.9, 0.95, 1.0), clamp(aInfo.y, 0, 1).mul(0.55));
      const alpha = select(isShadow, shadowA.mul(aTint.a), col.a.mul(aTint.a));
      spriteMat.colorNode = vec4(flashed, alpha);
    }
    spriteMat.transparent = true;
    spriteMat.depthWrite = false;
    spriteMat.depthTest = true;
    spriteMat.side = THREE.DoubleSide;

    this.sprites = new THREE.InstancedMesh(geo, spriteMat, MAX_SPRITES);
    this.sprites.frustumCulled = false;
    this.sprites.renderOrder = 1;
    this.sprites.count = 0;
    this.sprites.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    // ---- particles (screen-space glows, always on top)
    const ogeo = new THREE.PlaneGeometry(1, 1);
    this.oTint = new THREE.InstancedBufferAttribute(new Float32Array(MAX_PARTICLES * 4), 4);
    this.oTint.setUsage(THREE.DynamicDrawUsage);
    ogeo.setAttribute('oTint', this.oTint);
    const omat = new THREE.MeshBasicNodeMaterial();
    {
      const t = attribute('oTint', 'vec4');
      const d = length(uv().mul(2).sub(1));
      omat.colorNode = vec4(t.rgb, float(1).sub(smoothstep(0.0, 1.0, d)).mul(t.a));
    }
    omat.transparent = true;
    omat.depthTest = false;
    omat.depthWrite = false;
    omat.side = THREE.DoubleSide;
    this.overlay = new THREE.InstancedMesh(ogeo, omat, MAX_PARTICLES);
    this.overlay.frustumCulled = false;
    this.overlay.renderOrder = 10;
    this.overlay.count = 0;
    this.overlay.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    // ---- vignette over the design plane
    const vmat = new THREE.MeshBasicNodeMaterial();
    {
      const r = length(uv().mul(2).sub(1));
      vmat.colorNode = vec4(0.005, 0.01, 0.04, smoothstep(0.4, 1.3, r).mul(0.82));
    }
    vmat.transparent = true;
    vmat.depthTest = false;
    vmat.depthWrite = false;
    this.vignette = new THREE.Mesh(new THREE.PlaneGeometry(WORLD.x, WORLD.y), vmat);
    this.vignette.frustumCulled = false;
    this.vignette.renderOrder = 20;

    // ---- room mesh material (vertex colours + night lighting + cut-away)
    const roomMat = new THREE.MeshBasicNodeMaterial();
    {
      const P = positionWorld;
      const N = normalWorld;
      const emissive = attribute('emissive', 'float');
      const base = vertexColor().rgb;
      const lit = lightingN(P, N);
      const lighted = base.mul(min(lit, vec3(1.3, 1.3, 1.3)));
      const fire = mix(lighted, base.mul(1.6), emissive);
      // faces turned away from the camera are only seen through the cut: dark stone
      const toCam = vec3(0, STAGE.sinP, STAGE.cosP);
      const away = dot(N, toCam).lessThan(0);
      const cap = vec3(0.12, 0.11, 0.13).mul(min(lighting(vec3(P.x, 30, P.z)), vec3(1, 1, 1)));
      roomMat.colorNode = vec4(select(away, cap, fire), 1);
      roomMat.side = THREE.DoubleSide;
      // cut-away: nothing above the clip height
      (roomMat as any).clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, -1, 0), STAGE.clipUnits * STAGE.scale)];
    }
    this.room = new THREE.Mesh(new THREE.BufferGeometry(), roomMat);
    this.room.frustumCulled = false;
    this.room.renderOrder = 0;
    this.room.visible = false;

    // ---- graveyard ground
    const gmat = new THREE.MeshBasicNodeMaterial();
    {
      const wp = vec2(positionWorld.x.add(WORLD.x / 2), positionWorld.z.add(WORLD.y / 2));
      const g0 = texture(this.groundTex).sample(wp.div(576)).rgb.mul(1.5);
      const grey = dot(g0, vec3(0.3, 0.5, 0.2));
      const graded = mix(g0, vec3(grey, grey, grey), 0.25).mul(vec3(0.85, 1.0, 1.05));
      const L = lighting(vec3(wp.x, 0, wp.y));
      gmat.colorNode = vec4(graded.mul(L), 1);
    }
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(6000, 6000), gmat);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.frustumCulled = false;
    this.ground.renderOrder = 0;

    this.scene.add(this.room, this.ground, this.sprites, this.overlay, this.vignette);
  }

  // MARK: stage + camera
  private applyStage() {
    const g = this.game;
    this.shownStage = g.stageVersion;
    const mesh = g.room;
    this.flatMode = !mesh;
    this.room.visible = !!mesh;
    this.ground.visible = !mesh;
    if (mesh) {
      let geo = this.roomGeo.get(mesh);
      if (!geo) {
        geo = new THREE.BufferGeometry();
        const ib = new THREE.InterleavedBuffer(mesh.vertices, ROOM_VERTEX_FLOATS);
        geo.setAttribute('position', new THREE.InterleavedBufferAttribute(ib, 3, 0));
        geo.setAttribute('normal', new THREE.InterleavedBufferAttribute(ib, 3, 3));
        geo.setAttribute('color', new THREE.InterleavedBufferAttribute(ib, 3, 6));
        geo.setAttribute('emissive', new THREE.InterleavedBufferAttribute(ib, 1, 9));
        this.roomGeo.set(mesh, geo);
      }
      this.room.geometry = geo;
      this.room.scale.setScalar(STAGE.scale);
      this.room.position.set(0, -mesh.floorY * STAGE.scale, 0);
    }
    const a = g.spec.ambient;
    (this.ambient as any).value.setRGB(a[0], a[1], a[2]);
  }

  /** camera basis: right, up (screen-up in world), toward-camera */
  private basis() {
    const P = this.flatMode ? Math.PI / 2 : STAGE.pitch;
    const s = Math.sin(P), c = Math.cos(P);
    return {
      r: new THREE.Vector3(1, 0, 0),
      u: new THREE.Vector3(0, c, -s),
      n: new THREE.Vector3(0, s, c),
      P,
    };
  }

  private updateCamera(shake: { x: number; y: number }) {
    const { u, n } = this.basis();
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    const va = w / h;
    const pa = WORLD.x / WORLD.y;
    const width = va > pa ? WORLD.y * va : WORLD.x;
    const height = va > pa ? WORLD.y : WORLD.x / va;
    const T = this.flatMode ? WORLD.y / 2 : STAGE.centerY; // plane y of the look-at point
    const top = T + (height - WORLD.y) / 2 + shake.y;
    const cam = this.camera;
    cam.left = -width / 2 - shake.x;
    cam.right = width / 2 - shake.x;
    cam.top = top;
    cam.bottom = top - height;
    cam.position.set(0, 0, 0).addScaledVector(n, 3000);
    cam.up.copy(u);
    cam.lookAt(0, 0, 0);
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld(true);
  }

  private onResize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight));
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(w, h, false);
  }

  // MARK: frame
  frame() {
    const now = performance.now();
    this.game.step((now - this.last) / 1000);
    this.last = now;
    this.fpsFrames++;
    if (now - this.fpsT > 500) {
      this.fps = (this.fpsFrames * 1000) / (now - this.fpsT);
      this.fpsFrames = 0;
      this.fpsT = now;
    }
    if (this.game.loading) return;
    if (this.shownStage !== this.game.stageVersion) this.applyStage();

    const { r, u, n, P } = this.basis();
    const flat = this.flatMode;
    const T = flat ? WORLD.y / 2 : STAGE.centerY;
    const ox = flat ? WORLD.x / 2 : 0; // plane -> world offsets for the graveyard
    const oz = flat ? WORLD.y / 2 : 0;
    const sinP = Math.sin(P);

    // lights
    const lights: LightRec[] = this.game.lights();
    const A = (this.lightA as any).array as THREE.Vector4[];
    const C = (this.lightC as any).array as THREE.Vector4[];
    for (let i = 0; i < LIGHTS; i++) {
      const l = lights[i];
      if (l) {
        A[i].set(l.x - ox, l.z - oz, Math.max(1, l.radius), l.intensity);
        C[i].set(l.color[0], l.color[1], l.color[2], l.height);
      } else {
        A[i].set(0, 0, 1, 0);
        C[i].set(0, 0, 0, 0);
      }
    }

    // sprites
    const { sprites, particles } = this.game.buildScene();
    const m = this.sprites.instanceMatrix.array as Float32Array;
    const rect = this.aRect.array as Float32Array;
    const tint = this.aTint.array as Float32Array;
    const info = this.aInfo.array as Float32Array;
    const gp = this.aGpos.array as Float32Array;
    let count = 0;
    for (const s of sprites) {
      if (count >= MAX_SPRITES) break;
      const i = count++;
      // world position
      const wx = s.floor.x - ox;
      const wz = s.floor.y - oz;
      const sx = s.size.x;
      const sy = s.size.y;
      const o = i * 16;
      if (s.decal) {
        // lies on the floor; plane +Y points away from the camera (world -z)
        // size.y is already foreshortened (screen units), so the world depth is size.y / sin(pitch)
        const py = flat ? 0.5 : 1.5;
        m[o] = sx; m[o + 1] = 0; m[o + 2] = 0; m[o + 3] = 0;
        m[o + 4] = 0; m[o + 5] = 0; m[o + 6] = -sy / sinP; m[o + 7] = 0;
        m[o + 8] = 0; m[o + 9] = 1; m[o + 10] = 0; m[o + 11] = 0;
        m[o + 12] = wx; m[o + 13] = py; m[o + 14] = wz; m[o + 15] = 1;
      } else {
        // camera-facing billboard standing at the feet, lifted along screen-up
        const px = wx + u.x * s.lift;
        const py = u.y * s.lift + (flat ? 1 : 0); // graveyard: sit just above the ground quad
        const pz = wz + u.z * s.lift;
        const cr = Math.cos(s.rot), sr = Math.sin(s.rot);
        // X' = (r*cos + u*sin) * sx ; Y' = (-r*sin + u*cos) * sy ; Z' = n
        m[o] = (r.x * cr + u.x * sr) * sx; m[o + 1] = (r.y * cr + u.y * sr) * sx; m[o + 2] = (r.z * cr + u.z * sr) * sx; m[o + 3] = 0;
        m[o + 4] = (-r.x * sr + u.x * cr) * sy; m[o + 5] = (-r.y * sr + u.y * cr) * sy; m[o + 6] = (-r.z * sr + u.z * cr) * sy; m[o + 7] = 0;
        m[o + 8] = n.x; m[o + 9] = n.y; m[o + 10] = n.z; m[o + 11] = 0;
        m[o + 12] = px; m[o + 13] = py; m[o + 14] = pz; m[o + 15] = 1;
      }
      // atlas rect = source texture rect composed with the frame sub-rect
      let ar: [number, number, number, number] = [0, 0, 1, 1];
      if (s.tex) ar = this.atlasRect.get(s.tex) ?? ar;
      rect[i * 4] = ar[0] + s.uv[0] * ar[2];
      rect[i * 4 + 1] = ar[1] + s.uv[1] * ar[3];
      rect[i * 4 + 2] = s.uv[2] * ar[2];
      rect[i * 4 + 3] = s.uv[3] * ar[3];
      tint[i * 4] = s.tint[0]; tint[i * 4 + 1] = s.tint[1]; tint[i * 4 + 2] = s.tint[2]; tint[i * 4 + 3] = s.tint[3];
      info[i * 4] = s.kind; info[i * 4 + 1] = s.flash; info[i * 4 + 2] = s.smooth ? 1 : 0; info[i * 4 + 3] = s.lit ? 1 : 0;
      gp[i * 4] = s.gpos.x - ox; gp[i * 4 + 1] = s.gpos.y - oz; gp[i * 4 + 2] = 0; gp[i * 4 + 3] = 0;
    }
    this.sprites.count = count;
    this.sprites.instanceMatrix.needsUpdate = true;
    for (const a of [this.aRect, this.aTint, this.aInfo, this.aGpos]) a.needsUpdate = true;

    // particles: placed in the camera's screen plane through the world origin
    const om = this.overlay.instanceMatrix.array as Float32Array;
    const ot = this.oTint.array as Float32Array;
    let pc = 0;
    for (const p of particles) {
      if (pc >= MAX_PARTICLES) break;
      const i = pc++;
      const o = i * 16;
      const px = r.x * (p.x - STAGE.centerX) + u.x * (T - p.y);
      const py = r.y * (p.x - STAGE.centerX) + u.y * (T - p.y);
      const pz = r.z * (p.x - STAGE.centerX) + u.z * (T - p.y);
      om[o] = r.x * p.size; om[o + 1] = r.y * p.size; om[o + 2] = r.z * p.size; om[o + 3] = 0;
      om[o + 4] = u.x * p.size; om[o + 5] = u.y * p.size; om[o + 6] = u.z * p.size; om[o + 7] = 0;
      om[o + 8] = n.x; om[o + 9] = n.y; om[o + 10] = n.z; om[o + 11] = 0;
      om[o + 12] = px; om[o + 13] = py; om[o + 14] = pz; om[o + 15] = 1;
      ot[i * 4] = p.color[0]; ot[i * 4 + 1] = p.color[1]; ot[i * 4 + 2] = p.color[2]; ot[i * 4 + 3] = p.alpha;
    }
    this.overlay.count = pc;
    this.overlay.instanceMatrix.needsUpdate = true;
    this.oTint.needsUpdate = true;

    // vignette: covers the 1280x800 design plane, facing the camera
    this.vignette.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, u, n));
    this.vignette.position
      .set(0, 0, 0)
      .addScaledVector(u, T - WORLD.y / 2)
      .addScaledVector(n, -50);

    this.updateCamera(this.game.shake);
    this.renderer.render(this.scene, this.camera);
  }

  get elapsedSeconds() {
    return (performance.now() - this.startTime) / 1000;
  }
}
