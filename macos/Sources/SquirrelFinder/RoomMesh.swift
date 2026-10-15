import Foundation
import ImageIO
import simd

/// Mirrors `RVert` in the Metal source (10 packed floats, stride 40).
struct RoomVertex {
    var x: Float, y: Float, z: Float
    var nx: Float, ny: Float, nz: Float
    var r: Float, g: Float, b: Float
    var emissive: Float
}

/// A MagicaVoxel OBJ room (palette-textured, 0.1 unit voxels) plus what the game needs from it:
/// a walkability grid derived from the surface heights and the glowing voxels as light sources.
final class RoomMesh {
    static let voxel: Float = 0.1
    static let gridN = 50                         // rooms span x, z in [-2.5, 2.5]

    private(set) var vertices: [RoomVertex] = []
    private(set) var floorY: Float = 1
    /// true = cannot stand here (wall, furniture, pit, water, outside the room).
    private(set) var blocked: [Bool]
    /// Brightest warm-emissive clusters (torches, lava): position in room units, rgb colour, voxel count.
    private(set) var glows: [(pos: SIMD3<Float>, color: SIMD3<Float>, weight: Int)] = []

    init?(objURL: URL, paletteURL: URL) {
        let n = RoomMesh.gridN
        blocked = [Bool](repeating: true, count: n * n)

        guard let palette = RoomMesh.loadPalette(paletteURL) else {
            NSLog("RoomMesh: palette failed: \(paletteURL.path)")
            return nil
        }
        guard let text = try? String(contentsOf: objURL, encoding: .utf8) else {
            NSLog("RoomMesh: obj unreadable: \(objURL.path)")
            return nil
        }

        var pos: [SIMD3<Float>] = []
        var nor: [SIMD3<Float>] = []
        var uvs: [Float] = []                    // only u matters: the palette is 256 x 1
        pos.reserveCapacity(60_000)

        // Palette entries that read as fire: bright, saturated, red-orange.
        var emissiveEntry = [Bool](repeating: false, count: 256)
        for i in 0..<256 {
            let c = palette[i]
            let mx = max(c.x, c.y, c.z), mn = min(c.x, c.y, c.z)
            let sat = mx > 0 ? (mx - mn) / mx : 0
            emissiveEntry[i] = mx > 0.78 && sat > 0.6 && c.x >= c.y && c.y < 0.7
        }

        var verts: [RoomVertex] = []
        var tops: [(x0: Float, x1: Float, z0: Float, z1: Float, y: Float)] = []
        var emissiveAcc: [SIMD3<Int32>: (sum: SIMD3<Float>, col: SIMD3<Float>, n: Int)] = [:]

        for sub in text.split(whereSeparator: \.isNewline) {          // OBJ files here use CRLF
            let s = sub.utf8
            guard let first = s.first else { continue }
            if first == UInt8(ascii: "v") {
                let parts = sub.split(separator: " ")
                switch parts[0] {
                case "v": pos.append(SIMD3(Float(parts[1])!, Float(parts[2])!, Float(parts[3])!))
                case "vn": nor.append(SIMD3(Float(parts[1])!, Float(parts[2])!, Float(parts[3])!))
                case "vt": uvs.append(Float(parts[1])!)
                default: break
                }
            } else if first == UInt8(ascii: "f") {
                let parts = sub.split(separator: " ")
                guard parts.count >= 4 else { continue }
                var corners: [(Int, Int, Int)] = []
                for q in parts[1...] {
                    let f = q.split(separator: "/")
                    corners.append((Int(f[0])! - 1, Int(f[1])! - 1, Int(f[2])! - 1))
                }
                for k in 1..<(corners.count - 1) {
                    let tri = [corners[0], corners[k], corners[k + 1]]
                    let nrm = nor[tri[0].2]
                    let entry = min(255, Int(uvs[tri[0].1] * 256))
                    let col = palette[entry]                  // sRGB; the view is sRGB so vertices carry linear
                    let lin = SIMD3<Float>(pow(col.x, 2.2), pow(col.y, 2.2), pow(col.z, 2.2))
                    let em: Float = emissiveEntry[entry] ? 1 : 0
                    var cen = SIMD3<Float>.zero
                    for c in tri {
                        let p = pos[c.0]
                        verts.append(RoomVertex(x: p.x, y: p.y, z: p.z, nx: nrm.x, ny: nrm.y, nz: nrm.z,
                                                r: lin.x, g: lin.y, b: lin.z, emissive: em))
                        cen += p / 3
                    }
                    if nrm.y > 0.9 {
                        let xs = tri.map { pos[$0.0].x }, zs = tri.map { pos[$0.0].z }
                        tops.append((xs.min()!, xs.max()!, zs.min()!, zs.max()!, pos[tri[0].0].y))
                    }
                    if em > 0 {
                        let key = SIMD3<Int32>(Int32((cen.x / 0.6).rounded(.down)),
                                               Int32((cen.y / 0.6).rounded(.down)),
                                               Int32((cen.z / 0.6).rounded(.down)))
                        var e = emissiveAcc[key] ?? (.zero, .zero, 0)
                        e.sum += cen; e.col += col; e.n += 1
                        emissiveAcc[key] = e
                    }
                }
            }
        }
        guard !verts.isEmpty else {
            NSLog("RoomMesh: no faces parsed")
            return nil
        }
        vertices = verts

        // Surface height per cell (max top-face y); cells with no top face stay at -inf (pit / outside).
        var height = [Float](repeating: -.infinity, count: n * n)
        let half = Float(n) / 2
        for t in tops {
            let ix0 = max(0, Int(((t.x0 / RoomMesh.voxel) + half).rounded()))
            let ix1 = min(n, Int(((t.x1 / RoomMesh.voxel) + half).rounded()))
            let iz0 = max(0, Int(((t.z0 / RoomMesh.voxel) + half).rounded()))
            let iz1 = min(n, Int(((t.z1 / RoomMesh.voxel) + half).rounded()))
            guard ix1 > ix0, iz1 > iz0 else { continue }
            for iz in iz0..<iz1 { for ix in ix0..<ix1 { height[iz * n + ix] = max(height[iz * n + ix], t.y) } }
        }
        // Floor level = the most common surface height (rounded to a voxel).
        var hist: [Int: Int] = [:]
        for h in height where h.isFinite { hist[Int((h * 10).rounded()), default: 0] += 1 }
        let mode = hist.max { $0.value < $1.value }?.key ?? 10
        floorY = Float(mode) / 10
        for i in 0..<(n * n) {
            let h = height[i]
            blocked[i] = !h.isFinite || h >= floorY + 0.35 || h <= floorY - 0.35
        }

        glows = emissiveAcc.values
            .sorted { $0.n > $1.n }
            .prefix(8)
            .map { e in
                let c = e.col / Float(e.n)
                let lin = SIMD3<Float>(pow(c.x, 2.2), pow(c.y, 2.2), pow(c.z, 2.2))
                return (pos: e.sum / Float(e.n), color: simd_clamp(lin * 1.4, SIMD3(repeating: 0), SIMD3(repeating: 1)), weight: e.n)
            }
    }

    private static func loadPalette(_ url: URL) -> [SIMD3<Float>]? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let w = img.width
        guard w >= 1, img.height >= 1 else { return nil }
        var px = [UInt8](repeating: 0, count: w * 4)
        guard let ctx = CGContext(data: &px, width: w, height: 1, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: 1))
        var out = [SIMD3<Float>](repeating: .zero, count: 256)
        for i in 0..<min(256, w) {
            out[i] = SIMD3(Float(px[i * 4]), Float(px[i * 4 + 1]), Float(px[i * 4 + 2])) / 255
        }
        return out
    }
}
