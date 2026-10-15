#!/usr/bin/env python3
"""Tiny software renderer for MagicaVoxel OBJ rooms, to eyeball geometry. usage: voxel_preview.py room_dir out.png [yaw pitch]"""
import glob, math, sys
import numpy as np
from PIL import Image

def load(d):
    obj = glob.glob(f"{d}/Package/*.obj")[0]
    png = glob.glob(f"{d}/Package/*.png")[0]
    pal = np.array(Image.open(png).convert("RGB"))[0]          # 256 x 3
    v, vt, vn, tris = [], [], [], []
    for line in open(obj):
        p = line.split()
        if not p: continue
        if p[0] == "v": v.append([float(x) for x in p[1:4]])
        elif p[0] == "vt": vt.append([float(x) for x in p[1:3]])
        elif p[0] == "vn": vn.append([float(x) for x in p[1:4]])
        elif p[0] == "f":
            idx = [tuple(int(i) for i in q.split("/")) for q in p[1:]]
            for k in range(1, len(idx) - 1): tris.append((idx[0], idx[k], idx[k + 1]))
    return np.array(v), np.array(vt), np.array(vn), tris, pal

def render(d, yaw=30, pitch=40, W=520, H=520):
    v, vt, vn, tris, pal = load(d)
    ya, pa = math.radians(yaw), math.radians(pitch)
    Ry = np.array([[math.cos(ya), 0, math.sin(ya)], [0, 1, 0], [-math.sin(ya), 0, math.cos(ya)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(pa), -math.sin(pa)], [0, math.sin(pa), math.cos(pa)]])
    R = Rx @ Ry
    c = np.array([0, 2.5, 0])
    pts = (v - c) @ R.T
    s = 78
    sx = W / 2 + pts[:, 0] * s
    sy = H / 2 - pts[:, 1] * s
    sz = pts[:, 2]
    img = np.zeros((H, W, 3), np.uint8) + 18
    zb = np.full((H, W), -1e9)
    light = np.array([-0.4, 0.8, 0.5]); light /= np.linalg.norm(light)
    for a, b, cc in tris:
        ia, ib, ic = a[0] - 1, b[0] - 1, cc[0] - 1
        n = vn[a[2] - 1]
        col = pal[min(255, int(vt[a[1] - 1][0] * 256))].astype(float)
        sh = 0.55 + 0.45 * max(0.0, float(n @ light))
        col = np.clip(col * sh, 0, 255).astype(np.uint8)
        x0, x1 = sx[[ia, ib, ic]].min(), sx[[ia, ib, ic]].max()
        y0, y1 = sy[[ia, ib, ic]].min(), sy[[ia, ib, ic]].max()
        xi0, xi1 = max(0, int(x0)), min(W - 1, int(x1) + 1)
        yi0, yi1 = max(0, int(y0)), min(H - 1, int(y1) + 1)
        if xi1 < xi0 or yi1 < yi0: continue
        X, Y = np.meshgrid(np.arange(xi0, xi1 + 1) + 0.5, np.arange(yi0, yi1 + 1) + 0.5)
        ax, ay, bx, by, cx, cy = sx[ia], sy[ia], sx[ib], sy[ib], sx[ic], sy[ic]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-9: continue
        w0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / den
        w1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / den
        w2 = 1 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any(): continue
        z = w0 * sz[ia] + w1 * sz[ib] + w2 * sz[ic]
        sub = zb[yi0:yi1 + 1, xi0:xi1 + 1]
        upd = m & (z > sub)
        sub[upd] = z[upd]
        img[yi0:yi1 + 1, xi0:xi1 + 1][upd] = col
    return Image.fromarray(img)

if __name__ == "__main__":
    d, out = sys.argv[1], sys.argv[2]
    yaw = float(sys.argv[3]) if len(sys.argv) > 3 else 30
    pitch = float(sys.argv[4]) if len(sys.argv) > 4 else 40
    render(d, yaw, pitch).save(out)
