import ImageIO
import MetalKit
import SwiftUI

final class Renderer: NSObject, MTKViewDelegate {
    /// Mirrors `Uniforms` in the Metal source (stride 80).
    struct Uniforms {
        var scale: V2
        var offset: V2
        var worldSize: V2
        var time: Float
        var lightCount: Float = 0
        var cam: SIMD4<Float> = .zero      // sin(pitch), cos(pitch), floor-centre plane y, clip height (game units)
        var cam2: SIMD4<Float> = .zero     // room units -> game units, floor y, depth gain, plane centre x
        var ambient: SIMD4<Float> = .zero
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let spritePipeline: MTLRenderPipelineState
    private let roomPipeline: MTLRenderPipelineState
    private let roomDepth: MTLDepthStencilState
    private let spriteDepth: MTLDepthStencilState
    private var textures: [TexID: MTLTexture] = [:]
    private let game: Game

    private let maxInstances = 4096
    private var buffers: [MTLBuffer] = []
    private var roomBuffers: [Int: (buffer: MTLBuffer, count: Int)] = [:]
    private var frame = 0
    private let inflight = DispatchSemaphore(value: 3)
    private var last = CACurrentMediaTime()
    private let startTime = CACurrentMediaTime()

    init?(view: MTKView, game: Game) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            NSLog("Metal device unavailable")
            return nil
        }
        self.device = device
        self.queue = queue
        self.game = game

        view.device = device
        view.colorPixelFormat = .bgra8Unorm_srgb
        view.depthStencilPixelFormat = .depth32Float
        view.clearDepth = 1.0
        view.clearColor = MTLClearColor(red: 0.004, green: 0.008, blue: 0.02, alpha: 1)
        view.preferredFramesPerSecond = 120
        view.framebufferOnly = ProcessInfo.processInfo.environment["SF_SNAPSHOT"] == nil

        do {
            let lib = try device.makeLibrary(source: shaderSource, options: nil)

            let sd = MTLRenderPipelineDescriptor()
            sd.vertexFunction = lib.makeFunction(name: "sprite_vs")
            sd.fragmentFunction = lib.makeFunction(name: "sprite_fs")
            sd.depthAttachmentPixelFormat = .depth32Float
            let ca = sd.colorAttachments[0]!
            ca.pixelFormat = view.colorPixelFormat
            ca.isBlendingEnabled = true
            ca.rgbBlendOperation = .add
            ca.alphaBlendOperation = .add
            ca.sourceRGBBlendFactor = .sourceAlpha
            ca.sourceAlphaBlendFactor = .one
            ca.destinationRGBBlendFactor = .oneMinusSourceAlpha
            ca.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            spritePipeline = try device.makeRenderPipelineState(descriptor: sd)

            let rd = MTLRenderPipelineDescriptor()
            rd.vertexFunction = lib.makeFunction(name: "room_vs")
            rd.fragmentFunction = lib.makeFunction(name: "room_fs")
            rd.depthAttachmentPixelFormat = .depth32Float
            rd.colorAttachments[0].pixelFormat = view.colorPixelFormat
            roomPipeline = try device.makeRenderPipelineState(descriptor: rd)
        } catch {
            NSLog("Metal pipeline error: \(error)")
            return nil
        }

        let rdState = MTLDepthStencilDescriptor()
        rdState.depthCompareFunction = .less
        rdState.isDepthWriteEnabled = true
        let sdState = MTLDepthStencilDescriptor()
        sdState.depthCompareFunction = .lessEqual
        sdState.isDepthWriteEnabled = false
        guard let a = device.makeDepthStencilState(descriptor: rdState),
              let b = device.makeDepthStencilState(descriptor: sdState) else {
            NSLog("Depth stencil state creation failed")
            return nil
        }
        roomDepth = a
        spriteDepth = b

        let loader = MTKTextureLoader(device: device)
        for id in TexID.allCases {
            guard let url = Game.assetDir(id.file + ".png") else {
                NSLog("Missing asset \(id.file).png")
                return nil
            }
            let opts: [MTKTextureLoader.Option: Any] = [
                .SRGB: true,
                .generateMipmaps: false,
                .textureUsage: MTLTextureUsage.shaderRead.rawValue,
                .textureStorageMode: MTLStorageMode.private.rawValue,
                .origin: MTKTextureLoader.Origin.topLeft,
            ]
            do { textures[id] = try loader.newTexture(URL: url, options: opts) } catch {
                NSLog("Texture load failed for \(id.file): \(error)")
                return nil
            }
        }

        for _ in 0..<3 {
            buffers.append(device.makeBuffer(length: MemoryLayout<SpriteInstance>.stride * maxInstances,
                                             options: .storageModeShared)!)
        }
        super.init()
        // Headless snapshots: don't depend on the display link (it stops when the screen is asleep/occluded).
        if ProcessInfo.processInfo.environment["SF_SNAPSHOT"] != nil {
            view.isPaused = true
            view.enableSetNeedsDisplay = false
            Timer.scheduledTimer(withTimeInterval: 1.0 / 120.0, repeats: true) { [weak view] _ in view?.draw() }
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    private func roomBuffer() -> (buffer: MTLBuffer, count: Int)? {
        guard let mesh = game.room else { return nil }
        let key = game.levelIndex
        if let cached = roomBuffers[key] { return cached }
        let verts = mesh.vertices
        guard !verts.isEmpty,
              let buf = device.makeBuffer(bytes: verts, length: MemoryLayout<RoomVertex>.stride * verts.count,
                                          options: .storageModeShared) else { return nil }
        let entry = (buffer: buf, count: verts.count)
        roomBuffers[key] = entry
        return entry
    }

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        game.step(Float(now - last))
        last = now

        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable else { return }

        inflight.wait()
        let scene = game.buildScene()
        let buf = buffers[frame % buffers.count]
        frame += 1
        let ptr = buf.contents().bindMemory(to: SpriteInstance.self, capacity: maxInstances)
        let count = min(scene.count, maxInstances)
        for i in 0..<count { ptr[i] = scene[i].inst }

        // aspect-fit the fixed-size plane into the drawable
        let size = view.drawableSize
        let worldAspect = Game.world.x / Game.world.y
        let viewAspect = Float(size.width / max(1, size.height))
        let fx = viewAspect > worldAspect ? worldAspect / viewAspect : 1
        let fy = viewAspect > worldAspect ? 1 : viewAspect / worldAspect
        let sc = V2(2 * fx / Game.world.x, -2 * fy / Game.world.y)
        var u = Uniforms(scale: sc, offset: V2(-fx, fy) + game.shake * sc, worldSize: Game.world,
                         time: Float(now - startTime))
        u.cam = SIMD4(StageCam.sinP, StageCam.cosP, StageCam.centerY, StageCam.clipUnits * StageCam.scale)
        u.cam2 = SIMD4(StageCam.scale, game.room?.floorY ?? 0, StageCam.depthK, StageCam.centerX)
        let amb = game.spec.ambient
        u.ambient = SIMD4(amb.x, amb.y, amb.z, 0)

        var lights = game.lights()
        u.lightCount = Float(lights.count)
        while lights.count < 32 { lights.append(LightRec(a: .zero, color: .zero)) }

        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else {
            inflight.signal()
            return
        }

        // 1) the voxel room, depth-tested
        if let room = roomBuffer() {
            enc.setRenderPipelineState(roomPipeline)
            enc.setDepthStencilState(roomDepth)
            enc.setCullMode(.none)
            enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
            enc.setVertexBuffer(room.buffer, offset: 0, index: 1)
            enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
            enc.setFragmentBytes(&lights, length: MemoryLayout<LightRec>.stride * 32, index: 2)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: room.count)
        }

        // 2) sprites, shadows, particles: depth-tested against the room, painter-ordered among themselves
        enc.setRenderPipelineState(spritePipeline)
        enc.setDepthStencilState(spriteDepth)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.setVertexBuffer(buf, offset: 0, index: 1)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.setFragmentBytes(&lights, length: MemoryLayout<LightRec>.stride * 32, index: 2)

        // consecutive items sharing a texture are drawn in one instanced call
        var runStart = 0
        var current: TexID = .ground
        var started = false
        func flush(_ end: Int) {
            guard started, end > runStart else { return }
            enc.setFragmentTexture(textures[current], index: 0)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4,
                               instanceCount: end - runStart, baseInstance: runStart)
        }
        for i in 0..<count {
            guard let t = scene[i].tex else { started = started || true; continue }
            if !started { current = t; started = true; runStart = 0 }
            if t != current {
                flush(i)
                runStart = i
                current = t
            }
        }
        flush(count)
        enc.endEncoding()

        cmd.addCompletedHandler { [inflight] _ in inflight.signal() }
        cmd.present(drawable)
        cmd.commit()

        // Debug aid: SF_SNAPSHOT=/path/out.png renders a frame (SF_SNAPSHOT_FRAME, default 150) and quits.
        if let out = ProcessInfo.processInfo.environment["SF_SNAPSHOT"],
           frame == (Int(ProcessInfo.processInfo.environment["SF_SNAPSHOT_FRAME"] ?? "") ?? 150) {
            cmd.waitUntilCompleted()
            Renderer.write(drawable.texture, to: out)
            exit(0)
        }
    }

    private static func write(_ tex: MTLTexture, to path: String) {
        let w = tex.width, h = tex.height, bpr = w * 4
        var bytes = [UInt8](repeating: 0, count: bpr * h)
        tex.getBytes(&bytes, bytesPerRow: bpr, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        let info = CGBitmapInfo.byteOrder32Little.union(CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue))
        guard let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: bpr,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info.rawValue),
              let img = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, img, nil)
        CGImageDestinationFinalize(dest)
    }
}

struct MetalView: NSViewRepresentable {
    let game: Game

    func makeCoordinator() -> Coordinator { Coordinator() }
    final class Coordinator { var renderer: Renderer? }

    func makeNSView(context: Context) -> MTKView {
        let v = MTKView()
        context.coordinator.renderer = Renderer(view: v, game: game)
        v.delegate = context.coordinator.renderer
        return v
    }

    func updateNSView(_ nsView: MTKView, context: Context) {}
}
