/**
 * A MagicaVoxel OBJ room (palette-textured, 0.1 unit voxels) plus what the game needs from it:
 * a walkability grid derived from the surface heights and the glowing voxels as light sources.
 * Port of macos/Sources/SquirrelFinder/RoomMesh.swift.
 */
export const VOXEL = 0.1;
export const GRID_N = 50; // rooms span x, z in [-2.5, 2.5]

/** floats per vertex: position(3) normal(3) colour(3) emissive(1) */
export const ROOM_VERTEX_FLOATS = 10;

export interface RoomGlow {
  pos: [number, number, number]; // room units
  color: [number, number, number]; // linear rgb
  weight: number;
}

export interface RoomMesh {
  vertices: Float32Array;
  vertexCount: number;
  floorY: number;
  /** true = cannot stand here (wall, furniture, pit, water, outside the room); index = z * GRID_N + x */
  blocked: Uint8Array;
  glows: RoomGlow[];
}

async function loadPalette(url: string): Promise<[number, number, number][]> {
  const blob = await (await fetch(url)).blob();
  const bmp = await createImageBitmap(blob, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  const w = Math.min(256, bmp.width);
  const canvas = new OffscreenCanvas(bmp.width, 1);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, 1).data;
  const out: [number, number, number][] = Array.from({ length: 256 }, () => [0, 0, 0] as [number, number, number]);
  for (let i = 0; i < w; i++) out[i] = [px[i * 4] / 255, px[i * 4 + 1] / 255, px[i * 4 + 2] / 255];
  return out;
}

export async function loadRoomMesh(objUrl: string, paletteUrl: string): Promise<RoomMesh> {
  const [palette, text] = await Promise.all([
    loadPalette(paletteUrl),
    fetch(objUrl).then((r) => r.text()),
  ]);

  // Palette entries that read as fire: bright, saturated, red-orange.
  const emissiveEntry = palette.map(([r, g, b]) => {
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    const sat = mx > 0 ? (mx - mn) / mx : 0;
    return mx > 0.78 && sat > 0.6 && r >= g && g < 0.7;
  });

  const pos: number[] = [];
  const nor: number[] = [];
  const uvU: number[] = [];
  const verts: number[] = [];
  const tops: { x0: number; x1: number; z0: number; z1: number; y: number }[] = [];
  const emissiveAcc = new Map<string, { sum: [number, number, number]; col: [number, number, number]; n: number }>();

  for (const line of text.split(/\r?\n/)) {
    if (line.length < 2) continue;
    const c0 = line.charCodeAt(0);
    if (c0 === 118 /* v */) {
      const parts = line.split(' ');
      switch (parts[0]) {
        case 'v': pos.push(+parts[1], +parts[2], +parts[3]); break;
        case 'vn': nor.push(+parts[1], +parts[2], +parts[3]); break;
        case 'vt': uvU.push(+parts[1]); break;
      }
    } else if (c0 === 102 /* f */) {
      const parts = line.split(' ');
      if (parts.length < 4) continue;
      const corners: [number, number, number][] = [];
      for (let i = 1; i < parts.length; i++) {
        const f = parts[i].split('/');
        corners.push([+f[0] - 1, +f[1] - 1, +f[2] - 1]);
      }
      for (let k = 1; k < corners.length - 1; k++) {
        const tri = [corners[0], corners[k], corners[k + 1]];
        const ni = tri[0][2] * 3;
        const nx = nor[ni], ny = nor[ni + 1], nz = nor[ni + 2];
        const entry = Math.min(255, Math.floor(uvU[tri[0][1]] * 256));
        const col = palette[entry];
        const lr = Math.pow(col[0], 2.2), lg = Math.pow(col[1], 2.2), lb = Math.pow(col[2], 2.2); // the canvas is sRGB
        const em = emissiveEntry[entry] ? 1 : 0;
        let cx = 0, cy = 0, cz = 0;
        for (const c of tri) {
          const pi = c[0] * 3;
          const px = pos[pi], py = pos[pi + 1], pz = pos[pi + 2];
          verts.push(px, py, pz, nx, ny, nz, lr, lg, lb, em);
          cx += px / 3; cy += py / 3; cz += pz / 3;
        }
        if (ny > 0.9) {
          const xs = tri.map((c) => pos[c[0] * 3]);
          const zs = tri.map((c) => pos[c[0] * 3 + 2]);
          tops.push({
            x0: Math.min(...xs), x1: Math.max(...xs),
            z0: Math.min(...zs), z1: Math.max(...zs),
            y: pos[tri[0][0] * 3 + 1],
          });
        }
        if (em > 0) {
          const key = `${Math.floor(cx / 0.6)},${Math.floor(cy / 0.6)},${Math.floor(cz / 0.6)}`;
          let e = emissiveAcc.get(key);
          if (!e) {
            e = { sum: [0, 0, 0], col: [0, 0, 0], n: 0 };
            emissiveAcc.set(key, e);
          }
          e.sum[0] += cx; e.sum[1] += cy; e.sum[2] += cz;
          e.col[0] += col[0]; e.col[1] += col[1]; e.col[2] += col[2];
          e.n++;
        }
      }
    }
  }
  if (verts.length === 0) throw new Error(`no faces parsed from ${objUrl}`);

  // Surface height per cell (max top-face y); cells with no top face stay at -inf (pit / outside).
  const n = GRID_N;
  const half = n / 2;
  const height = new Float32Array(n * n).fill(-Infinity);
  for (const t of tops) {
    const ix0 = Math.max(0, Math.round(t.x0 / VOXEL + half));
    const ix1 = Math.min(n, Math.round(t.x1 / VOXEL + half));
    const iz0 = Math.max(0, Math.round(t.z0 / VOXEL + half));
    const iz1 = Math.min(n, Math.round(t.z1 / VOXEL + half));
    if (ix1 <= ix0 || iz1 <= iz0) continue;
    for (let iz = iz0; iz < iz1; iz++) {
      for (let ix = ix0; ix < ix1; ix++) {
        const i = iz * n + ix;
        if (t.y > height[i]) height[i] = t.y;
      }
    }
  }
  // Floor level = the most common surface height (rounded to a voxel).
  const hist = new Map<number, number>();
  for (const h of height) if (Number.isFinite(h)) hist.set(Math.round(h * 10), (hist.get(Math.round(h * 10)) ?? 0) + 1);
  let mode = 10;
  let best = -1;
  for (const [k, v] of hist) if (v > best) { best = v; mode = k; }
  const floorY = mode / 10;
  const blocked = new Uint8Array(n * n);
  for (let i = 0; i < n * n; i++) {
    const h = height[i];
    blocked[i] = !Number.isFinite(h) || h >= floorY + 0.35 || h <= floorY - 0.35 ? 1 : 0;
  }

  const glows: RoomGlow[] = [...emissiveAcc.values()]
    .sort((a, b) => b.n - a.n)
    .slice(0, 8)
    .map((e) => {
      const lin = e.col.map((c) => Math.min(1, Math.pow(c / e.n, 2.2) * 1.4)) as [number, number, number];
      return {
        pos: [e.sum[0] / e.n, e.sum[1] / e.n, e.sum[2] / e.n],
        color: lin,
        weight: e.n,
      };
    });

  return { vertices: new Float32Array(verts), vertexCount: verts.length / ROOM_VERTEX_FLOATS, floorY, blocked, glows };
}
