import AppKit
import Combine
import simd

typealias V2 = SIMD2<Float>

enum GameState { case playing, won, lost }

/// Texture files in Assets/ (name = file stem).
enum TexID: String, CaseIterable {
    case ground, wizardWalk = "wizard_walk", wizardIdle = "wizard_idle", ghostWalk = "ghost_walk"
    case penguin, props, cracks
    var file: String { rawValue }
}

/// Shader-side primitive kinds (see Shaders.swift).
enum Kind: Float { case sprite = 0, shadow, glow, ring, ground, vignette, emissive }

/// Mirrors `Sprite` in the Metal source (stride 96).
struct SpriteInstance {
    var center: V2
    var size: V2
    var tint: SIMD4<Float>
    var rotation: Float = 0
    var kind: Float
    var param: Float = 0
    var seed: Float = 0                       // > 0.5: smooth (linear) sampling instead of pixel-art nearest
    var uv: SIMD4<Float> = SIMD4(0, 0, 1, 1)  // sub-rect of the texture: x, y, w, h
    var depthInfo: SIMD4<Float> = SIMD4(0.5, 0, 0, 0)   // depth at feet, feet plane y, depth slope, unused
    var gpos: SIMD4<Float> = .zero            // floor position in game units, for lighting
}

/// Mirrors `Light` in the Metal source (stride 32).
struct LightRec {
    var a: SIMD4<Float>       // x, z (game floor coords), radius, intensity
    var color: SIMD4<Float>   // rgb, height above the floor
}

struct SceneItem {
    var tex: TexID?          // nil = any texture (procedural kinds)
    var inst: SpriteInstance
}

struct Ghost {
    var p: V2
    var v: V2
    var age: Float = 0
    var phase: Float
    var near = false         // inside the near-miss radius right now (avoids double-counting)
}

struct Particle {
    var p: V2                // plane coordinates
    var v: V2
    var life: Float
    var maxLife: Float
    var size: Float
    var color: SIMD3<Float>
    var gravity: Float
    var grow: Float
}

struct DirProb: Identifiable {
    let id: String
    let label: String
    let p: Double
}

/// Camera for the voxel rooms: orthographic, tilted, looking at the floor centre.
enum StageCam {
    static let pitch: Float = 58 * .pi / 180
    static let sinP: Float = sin(pitch)
    static let cosP: Float = cos(pitch)
    static let scale: Float = 140          // game units per room unit (voxel = 14)
    static let centerX: Float = 640
    static let centerY: Float = 481        // plane y of the floor centre
    static let clipUnits: Float = 2.2      // room units of wall kept above the floor
    static let depthK: Float = 0.0004
}

final class Game: ObservableObject {
    // MARK: world
    static let world = V2(1280, 800)        // plane size (the flat graveyard plays on all of it)
    private let marginX: Float = 70
    private let marginTop: Float = 100
    private let marginBottom: Float = 70
    private let moveTime: Float = 0.35
    private let ghostArm: Float = 0.9           // seconds before a spawned ghost is dangerous
    private let propScaleBase: Float = 3

    // Scoring: +10 per second survived, +25 per near miss, +500 and a speed bonus for the penguin.
    private let pointsPerSecond: Float = 10
    private let nearMissPoints: Float = 25
    private let goalPoints: Float = 500

    /// Graveyard scenery from the Dark Horror pack (art drawn at 3 units per pixel unless `scale` says otherwise).
    /// `p` is the base-centre of the sprite; `glow` makes it a light source.
    private struct Prop {
        let name: String
        let p: V2
        var scale: Float = 3
        var flat = false                       // ground decal: always under characters
        var alpha: Float = 1
        var glow: (color: SIMD3<Float>, radius: Float, intensity: Float, lift: Float)?
    }
    private static let torchGlow = (color: SIMD3<Float>(1.0, 0.68, 0.30), radius: Float(330), intensity: Float(0.85), lift: Float(0.8))
    private static let mushroomGlow = (color: SIMD3<Float>(0.25, 1.0, 0.60), radius: Float(190), intensity: Float(0.65), lift: Float(0.5))
    private static let pumpkinGlow = (color: SIMD3<Float>(1.0, 0.50, 0.15), radius: Float(215), intensity: Float(0.70), lift: Float(0.5))
    private static let cauldronGlow = (color: SIMD3<Float>(0.40, 1.0, 0.30), radius: Float(290), intensity: Float(0.80), lift: Float(0.6))

    private let props: [Prop] = [
        Prop(name: "web", p: V2(120, 150), flat: true, alpha: 0.75),
        Prop(name: "web", p: V2(1215, 150), flat: true, alpha: 0.75),
        Prop(name: "blood", p: V2(800, 300), flat: true),
        Prop(name: "blood", p: V2(360, 400), flat: true),
        Prop(name: "skull", p: V2(760, 610), scale: 3.2, flat: true),
        Prop(name: "bones", p: V2(540, 545), scale: 3.2, flat: true),
        Prop(name: "crossBig", p: V2(150, 255)),
        Prop(name: "crossBlood", p: V2(1130, 205)),
        Prop(name: "treeDead", p: V2(95, 700)),
        Prop(name: "treeGreen", p: V2(1195, 500)),
        Prop(name: "tombstone", p: V2(330, 630)),
        Prop(name: "tombstone", p: V2(405, 665)),
        Prop(name: "tombstone", p: V2(930, 700)),
        Prop(name: "tombstone", p: V2(870, 170)),
        Prop(name: "crossSmall", p: V2(500, 150)),
        Prop(name: "crossSmall", p: V2(1085, 650)),
        Prop(name: "crossBroken", p: V2(245, 580)),
        Prop(name: "bushBig", p: V2(720, 120)),
        Prop(name: "bushBig", p: V2(1235, 340)),
        Prop(name: "bushSmall", p: V2(60, 520)),
        Prop(name: "bushSmall", p: V2(770, 715)),
        Prop(name: "stump", p: V2(430, 110)),
        Prop(name: "cauldron", p: V2(985, 135), glow: cauldronGlow),
        Prop(name: "torch", p: V2(120, 440), scale: 5, glow: torchGlow),
        Prop(name: "torch", p: V2(1170, 395), scale: 5, glow: torchGlow),
        Prop(name: "torch", p: V2(640, 765), scale: 5, glow: torchGlow),
        Prop(name: "torch", p: V2(580, 135), scale: 5, glow: torchGlow),
        Prop(name: "mushroomA", p: V2(195, 745), scale: 3.5, glow: mushroomGlow),
        Prop(name: "mushroomB", p: V2(1110, 575), scale: 3.5, glow: mushroomGlow),
        Prop(name: "mushroomC", p: V2(870, 770), scale: 3.5, glow: mushroomGlow),
        Prop(name: "pumpkin", p: V2(455, 725), scale: 3.5, glow: pumpkinGlow),
        Prop(name: "pumpkin", p: V2(1055, 700), scale: 3.5, glow: pumpkinGlow),
    ]

    // MARK: HUD state (published, updated sparingly)
    @Published var state: GameState = .playing
    @Published var shownTime: Double = 0
    @Published var score = 0
    @Published var bonus = 0                    // goal bonus shown on the end card
    @Published var bestScore: Int = UserDefaults.standard.integer(forKey: "sf_best_score")
    @Published var bestTime: Double? = UserDefaults.standard.object(forKey: "sf_best") as? Double
    @Published var levelIndex = 0
    @Published var levelName = ""
    @Published var levelTagline = ""
    @Published var models: [String] = []
    @Published var player: String = "" { didSet { if oldValue != player { restartRun() } } }
    @Published var aiStatus: String = ""
    @Published var aiThinking = false
    @Published var aiProbs: [DirProb] = []
    @Published var aiMoves = 0
    @Published var aiError: String?

    var levelCount: Int { Levels.all.count }
    var isLastLevel: Bool { levelIndex >= Levels.all.count - 1 }

    // MARK: level / stage
    private(set) var spec = Levels.all[0]
    private(set) var room: RoomMesh?
    private var meshCache: [String: RoomMesh] = [:]
    private var walk: [Bool] = []               // inflated walkability per grid cell
    private var spawnCells: [V2] = []           // cell centres (game units) in the main connected region
    private var levelGlows: [LightRec] = []
    private var levelStartScore: Float = 0
    private var loops = 0                       // completed passes through all levels (raises difficulty)
    private var startG = V2.zero

    var isRoom: Bool { room != nil }
    private var cellG: Float { RoomMesh.voxel * StageCam.scale }
    /// Sprites are drawn 1:1 with the art (128 px frames) in rooms, 1.5x in the flat graveyard.
    private var spriteSize: Float { isRoom ? 128 : 192 }
    private var rs: Float { spriteSize / 192 }   // radius / offset scale relative to the graveyard
    private var speedScale: Float { isRoom ? 0.55 : 1 }
    private var wizardSpeed: Float { 300 * speedScale }
    private var hitGhost: Float { 58 * rs }
    private var hitGoal: Float { 64 * rs }
    private var nearMissRadius: Float { 105 * rs }

    // MARK: simulation state
    private(set) var wizard = Game.world / 2
    private var ghosts: [Ghost] = []
    private var penguin: V2?
    private var penguinBorn: Float = 0
    private var particles: [Particle] = []
    private var elapsed: Float = 0
    private var scoreF: Float = 0
    private var clock: Float = 0                // animation clock, never frozen
    private var endClock: Float = 0
    private var penguinAt: Float = 2
    private var spawnTimer: Float = 1.5
    private var faceRow = 0                     // 0 = down; rows follow the sheets' 8-direction order
    private var walkPhase: Float = 0
    private var moving = false
    private var dustTimer: Float = 0
    private var lastShown: Int = -1
    private var trauma: Float = 0
    private(set) var shake = V2.zero            // plane units; the renderer adds it to the camera
    private var keys = Set<UInt16>()
    private var monitor: Any?

    // MARK: AI
    private let client = OllamaClient()
    private var aiDir = V2.zero
    private var moveLeft: Float = 0
    private var pending = false
    private var epoch = 0

    private static let dirs: [(key: String, label: String, vec: V2, desc: String)] = [
        ("up", "↑", V2(0, -1), "Walk straight up"),
        ("up_right", "↗", V2(1, -1), "Walk diagonally up and to the right"),
        ("right", "→", V2(1, 0), "Walk straight to the right"),
        ("down_right", "↘", V2(1, 1), "Walk diagonally down and to the right"),
        ("down", "↓", V2(0, 1), "Walk straight down"),
        ("down_left", "↙", V2(-1, 1), "Walk diagonally down and to the left"),
        ("left", "←", V2(-1, 0), "Walk straight to the left"),
        ("up_left", "↖", V2(-1, -1), "Walk diagonally up and to the left"),
    ]

    init() {
        if let p = ProcessInfo.processInfo.environment["SF_PLAYER"] { player = p }
        let start = Int(ProcessInfo.processInfo.environment["SF_LEVEL"] ?? "") ?? 1
        loadLevel(max(0, min(Levels.all.count - 1, start - 1)))
        installKeyMonitor()
        Task { await loadModels() }
    }

    deinit { if let m = monitor { NSEvent.removeMonitor(m) } }

    // MARK: levels
    private func loadLevel(_ index: Int) {
        levelIndex = index
        spec = Levels.all[index]
        levelName = spec.name
        levelTagline = spec.tagline
        room = nil
        walk = []
        spawnCells = []
        levelGlows = []
        if let slug = spec.room {
            if meshCache[slug] == nil, let dir = Game.assetDir("rooms") {
                meshCache[slug] = RoomMesh(objURL: dir.appendingPathComponent("\(slug).obj"),
                                           paletteURL: dir.appendingPathComponent("\(slug).png"))
            }
            if let m = meshCache[slug] {
                room = m
                prepareGrid(m)
            } else {
                NSLog("level \(index + 1) \(slug): mesh failed to load (assets: \(String(describing: Game.assetDir("rooms"))))")
            }
        }
        reset()
    }

    /// App bundle first, then the Assets folder next to Package.swift (dev builds).
    static func assetDir(_ sub: String) -> URL? {
        let fm = FileManager.default
        var dirs: [URL] = []
        if let r = Bundle.main.resourceURL { dirs.append(r.appendingPathComponent("Assets")) }
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        dirs.append(exe.deletingLastPathComponent().appendingPathComponent("Assets"))
        var src = URL(fileURLWithPath: #filePath)
        for _ in 0..<3 { src.deleteLastPathComponent() }
        dirs.append(src.appendingPathComponent("Assets"))
        for d in dirs {
            let u = d.appendingPathComponent(sub)
            if fm.fileExists(atPath: u.path) { return u }
        }
        return nil
    }

    private func prepareGrid(_ m: RoomMesh) {
        let n = RoomMesh.gridN
        // a cell is walkable when it and its 8 neighbours are free (keeps the wizard off the walls)
        var free = [Bool](repeating: false, count: n * n)
        for z in 1..<(n - 1) {
            for x in 1..<(n - 1) {
                var ok = true
                outer: for dz in -1...1 { for dx in -1...1 where m.blocked[(z + dz) * n + (x + dx)] { ok = false; break outer } }
                free[z * n + x] = ok
            }
        }
        // keep only the largest connected region so nothing spawns in sealed pockets
        var label = [Int](repeating: -1, count: n * n)
        var sizes: [Int] = []
        for start in 0..<(n * n) where free[start] && label[start] < 0 {
            let id = sizes.count
            var stack = [start]
            label[start] = id
            var count = 0
            while let c = stack.popLast() {
                count += 1
                let cx = c % n, cz = c / n
                for (nx, nz) in [(cx + 1, cz), (cx - 1, cz), (cx, cz + 1), (cx, cz - 1)]
                where nx >= 0 && nz >= 0 && nx < n && nz < n {
                    let ni = nz * n + nx
                    if free[ni] && label[ni] < 0 { label[ni] = id; stack.append(ni) }
                }
            }
            sizes.append(count)
        }
        let best = sizes.enumerated().max { $0.element < $1.element }?.offset ?? 0
        walk = (0..<(n * n)).map { label[$0] == best && best >= 0 && !sizes.isEmpty }
        spawnCells = []
        for i in 0..<(n * n) where walk[i] {
            let cx = Float(i % n) + 0.5 - Float(n) / 2, cz = Float(i / n) + 0.5 - Float(n) / 2
            spawnCells.append(V2(cx, cz) * cellG)
        }
        startG = spawnCells.min { simd_length($0) < simd_length($1) } ?? .zero

        let S = StageCam.scale
        levelGlows = m.glows.map { g in
            LightRec(a: SIMD4(g.pos.x * S, g.pos.z * S, 300, 0.95),
                     color: SIMD4(g.color.x, g.color.y, g.color.z, max(10, (g.pos.y - m.floorY) * S)))
        }
    }

    // MARK: lifecycle
    /// Restart the current level (retry after a loss, or space).
    func reset() {
        epoch += 1
        pending = false
        moveLeft = 0
        aiDir = .zero
        aiError = nil
        aiThinking = false
        aiProbs = []
        aiMoves = 0
        aiStatus = player.isEmpty ? "" : "\(player) is ready"

        wizard = isRoom ? startG : Game.world / 2
        penguin = nil
        particles = []
        elapsed = 0
        trauma = 0
        shake = .zero
        scoreF = levelStartScore
        score = Int(scoreF)
        bonus = 0
        endClock = 0
        penguinAt = .random(in: 1.0...3.0)
        spawnTimer = 2.0
        moving = false
        faceRow = 0
        state = .playing
        shownTime = 0
        lastShown = -1
        ghosts = (0..<spec.ghosts).map { _ in newGhost(armed: true) }
    }

    /// Start over from the first level (used when the player changes).
    private func restartRun() {
        levelStartScore = 0
        loops = 0
        loadLevel(0)
    }

    /// After a win: go to the next level, carrying the score.
    func nextLevel() {
        guard state == .won else { return }
        levelStartScore = scoreF
        if isLastLevel {
            loops += 1
            loadLevel(0)
        } else {
            loadLevel(levelIndex + 1)
        }
    }

    func loadModels() async {
        do {
            let list = try await client.models()
            await MainActor.run { self.models = list }
        } catch {
            await MainActor.run { self.aiStatus = error.localizedDescription }
        }
    }

    // MARK: input
    private func installKeyMonitor() {
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] e in
            guard let self else { return e }
            let handled: Set<UInt16> = [123, 124, 125, 126, 0, 1, 2, 13, 49]
            guard handled.contains(e.keyCode) else { return e }
            if e.type == .keyDown {
                if e.keyCode == 49 {
                    switch self.state {
                    case .won: self.nextLevel()
                    case .lost: self.reset()
                    case .playing: break
                    }
                } else { self.keys.insert(e.keyCode) }
            } else {
                self.keys.remove(e.keyCode)
            }
            return nil
        }
    }

    private func humanDirection() -> V2 {
        var d = V2.zero
        if keys.contains(123) || keys.contains(0) { d.x -= 1 }
        if keys.contains(124) || keys.contains(2) { d.x += 1 }
        if keys.contains(126) || keys.contains(13) { d.y -= 1 }
        if keys.contains(125) || keys.contains(1) { d.y += 1 }
        return d
    }

    // MARK: stage helpers
    private func cellIndex(_ g: V2) -> Int? {
        let n = RoomMesh.gridN
        let ix = Int((g.x / cellG + Float(n) / 2).rounded(.down))
        let iz = Int((g.y / cellG + Float(n) / 2).rounded(.down))
        guard ix >= 0, iz >= 0, ix < n, iz < n else { return nil }
        return iz * n + ix
    }

    private func isFree(_ g: V2) -> Bool {
        if isRoom {
            guard let i = cellIndex(g) else { return false }
            return walk[i]
        }
        return g.x >= marginX && g.x <= Game.world.x - marginX && g.y >= marginTop && g.y <= Game.world.y - marginBottom
    }

    private func slide(_ p: V2, by d: V2) -> V2 {
        let t = p + d
        if isFree(t) { return t }
        let tx = V2(p.x + d.x, p.y), ty = V2(p.x, p.y + d.y)
        if isFree(tx) { return tx }
        if isFree(ty) { return ty }
        return p
    }

    private func randomPoint(awayFrom p: V2, by d: Float) -> V2 {
        if isRoom {
            let far = spawnCells.filter { simd_distance($0, p) >= d }
            let c = far.randomElement() ?? spawnCells.max { simd_distance($0, p) < simd_distance($1, p) } ?? .zero
            return c + V2(.random(in: -4...4), .random(in: -4...4))
        }
        let w = Game.world
        for _ in 0..<60 {
            let c = V2(.random(in: marginX...(w.x - marginX)), .random(in: marginTop...(w.y - marginBottom)))
            if simd_distance(c, p) >= d { return c }
        }
        return V2(marginX, marginTop)
    }

    private func newGhost(armed: Bool) -> Ghost {
        let p = randomPoint(awayFrom: wizard, by: 260 * rs)
        let a = Float.random(in: 0..<(2 * .pi))
        let boost = 1 + 0.15 * Float(loops)
        let s = Float.random(in: spec.speed) * speedScale * boost
        return Ghost(p: p, v: V2(cos(a), sin(a)) * s, age: armed ? ghostArm : 0,
                     phase: .random(in: 0..<(2 * .pi)))
    }

    /// Row in the 8-direction sheets: down, down-left, left, up-left, up, up-right, right, down-right.
    private func dirRow(_ v: V2) -> Int {
        guard simd_length(v) > 0.001 else { return 0 }
        let a = atan2(v.x, v.y)                  // 0 = down, positive toward the right
        let idx = Int((-a / (.pi / 4)).rounded())
        return ((idx % 8) + 8) % 8
    }

    /// Floor position (game units) -> plane position of that floor point.
    private func plane(_ g: V2) -> V2 {
        isRoom ? V2(StageCam.centerX + g.x, StageCam.centerY + g.y * StageCam.sinP) : g
    }

    /// Where a character at floor position `g` is drawn: sprite centre and feet in plane coordinates.
    private func place(_ g: V2, feetOff: Float) -> (center: V2, feet: V2) {
        if isRoom {
            let f = plane(g)
            return (f - V2(0, feetOff), f)
        }
        return (g, g + V2(0, feetOff))
    }

    private func floorDepth(_ g: V2) -> Float {
        isRoom ? 0.55 - StageCam.depthK * (g.y * StageCam.cosP) : 0.5
    }

    // MARK: stepping
    func step(_ rawDt: Float) {
        let dt = min(rawDt, 1.0 / 30.0)
        clock += dt
        updateParticles(dt)
        trauma = max(0, trauma - 1.4 * dt)
        if trauma > 0 {
            let a = trauma * trauma * 30
            shake = V2(sin(clock * 91) + sin(clock * 53 + 1.3), cos(clock * 83) + sin(clock * 61 + 2.1)) * 0.5 * a
        } else {
            shake = .zero
        }

        guard state == .playing else {
            endClock += dt
            return
        }

        var dir = V2.zero
        if player.isEmpty {
            dir = humanDirection()
        } else {
            if aiError != nil { return }
            if moveLeft <= 0 {                  // wizard stands still while the model thinks; the world keeps moving
                moving = false
                if !pending { requestMove() }
            } else {
                dir = aiDir
                moveLeft -= dt
            }
        }
        elapsed += dt
        scoreF += pointsPerSecond * dt

        // wizard
        moving = simd_length(dir) > 0.01
        if moving {
            let n = simd_normalize(dir)
            wizard = slide(wizard, by: n * wizardSpeed * dt)
            faceRow = dirRow(n)
            walkPhase += dt * 11
            dustTimer -= dt
            if dustTimer <= 0 {
                dustTimer = 0.09
                emitDust(at: place(wizard, feetOff: 57 * rs).feet)
            }
        }

        // ghosts
        for i in ghosts.indices {
            ghosts[i].age += dt
            var p = ghosts[i].p
            let d = ghosts[i].v * dt
            let nx = V2(p.x + d.x, p.y)
            if isFree(nx) { p.x = nx.x } else { ghosts[i].v.x *= -1 }
            let ny = V2(p.x, p.y + d.y)
            if isFree(ny) { p.y = ny.y } else { ghosts[i].v.y *= -1 }
            ghosts[i].p = p

            // near miss: a dangerous ghost brushes past without touching
            if ghosts[i].age >= ghostArm {
                let dist = simd_distance(ghosts[i].p, wizard)
                if dist < nearMissRadius && dist >= hitGhost {
                    if !ghosts[i].near {
                        ghosts[i].near = true
                        scoreF += nearMissPoints
                        trauma = max(trauma, 0.2)
                        emitPuff(at: place(wizard, feetOff: 57 * rs).center + V2(0, -50 * rs), strength: 90)
                    }
                } else if dist > nearMissRadius + 40 * rs {
                    ghosts[i].near = false
                }
            }
        }
        spawnTimer -= dt
        if spawnTimer <= 0 && ghosts.count < spec.maxGhosts {
            ghosts.append(newGhost(armed: false))
            spawnTimer = max(1.0, spec.spawnEvery - elapsed * 0.04)
        }

        // penguin appears once
        if penguin == nil && elapsed >= penguinAt {
            let p = randomPoint(awayFrom: wizard, by: 340 * rs)
            penguin = p
            penguinBorn = clock
            let c = place(p, feetOff: 76 * rs).center
            for _ in 0..<9 { emitPuff(at: c, strength: 220) }
        }

        // HUD (time in tenths, score in whole points)
        let tenths = Int(elapsed * 10)
        if tenths != lastShown { lastShown = tenths; shownTime = Double(elapsed) }
        if Int(scoreF) != score { score = Int(scoreF) }

        // collisions
        for g in ghosts where g.age >= ghostArm && simd_distance(g.p, wizard) < hitGhost {
            finish(.lost, hit: g.p)
            return
        }
        if let s = penguin, simd_distance(s, wizard) < hitGoal { finish(.won, hit: s) }
    }

    private func finish(_ result: GameState, hit: V2) {
        state = result
        endClock = 0
        shownTime = Double(elapsed)
        epoch += 1                              // discard in-flight AI answers
        pending = false
        aiThinking = false
        if result == .won {
            bonus = Int(goalPoints) + Int(max(0, 60 - elapsed) * 10)    // quicker catch, bigger bonus
            scoreF += Float(bonus)
            if bestTime == nil || Double(elapsed) < bestTime! {
                bestTime = Double(elapsed)
                UserDefaults.standard.set(bestTime, forKey: "sf_best")
            }
            emitConfetti(at: place(hit, feetOff: 76 * rs).center)
        } else {
            emitSplat(at: place(wizard, feetOff: 57 * rs).center)
            trauma = 1                          // camera shake on being hit
        }
        score = Int(scoreF)
        if score > bestScore {
            bestScore = score
            UserDefaults.standard.set(bestScore, forKey: "sf_best_score")
        }
    }

    // MARK: particles (plane coordinates)
    private func emitDust(at p: V2) {
        particles.append(Particle(p: p + V2(.random(in: -8...8), 0),
                                  v: V2(.random(in: -25...25), .random(in: -30...(-8))),
                                  life: 0.5, maxLife: 0.5, size: .random(in: 12...20) * rs,
                                  color: SIMD3(0.45, 0.5, 0.62), gravity: 0, grow: 40))
    }

    private func emitPuff(at p: V2, strength: Float) {
        let a = Float.random(in: 0..<(2 * .pi))
        let s = Float.random(in: 0.3...1) * strength
        particles.append(Particle(p: p, v: V2(cos(a), sin(a)) * s, life: 0.7, maxLife: 0.7,
                                  size: .random(in: 18...34) * rs, color: SIMD3(0.9, 0.95, 1),
                                  gravity: 0, grow: 30))
    }

    private func emitConfetti(at p: V2) {
        let palette: [SIMD3<Float>] = [
            SIMD3(1.0, 0.35, 0.40), SIMD3(1.0, 0.82, 0.25), SIMD3(0.35, 0.85, 0.55),
            SIMD3(0.35, 0.65, 1.0), SIMD3(0.85, 0.5, 1.0),
        ]
        for _ in 0..<90 {
            let a = Float.random(in: 0..<(2 * .pi))
            let s = Float.random(in: 140...520)
            let life = Float.random(in: 0.9...1.8)
            particles.append(Particle(p: p, v: V2(cos(a), sin(a) * 0.9 - 0.5) * s, life: life,
                                      maxLife: life, size: .random(in: 9...16),
                                      color: palette.randomElement()!, gravity: 520, grow: 0))
        }
    }

    private func emitSplat(at p: V2) {
        for _ in 0..<40 {
            let a = Float.random(in: 0..<(2 * .pi))
            let s = Float.random(in: 80...380)
            let life = Float.random(in: 0.5...1.0)
            particles.append(Particle(p: p, v: V2(cos(a), sin(a)) * s, life: life, maxLife: life,
                                      size: .random(in: 10...20),
                                      color: SIMD3(0.75, 0.85, 1.0),
                                      gravity: 240, grow: -8))
        }
    }

    private func updateParticles(_ dt: Float) {
        guard !particles.isEmpty else { return }
        for i in particles.indices {
            particles[i].life -= dt
            particles[i].v.y += particles[i].gravity * dt
            particles[i].v *= (1 - 1.8 * dt)
            particles[i].p += particles[i].v * dt
            particles[i].size = max(0, particles[i].size + particles[i].grow * dt)
        }
        particles.removeAll { $0.life <= 0 }
    }

    // MARK: AI
    private func rel(_ target: V2) -> String {
        let d = target - wizard
        let h = abs(d.x) < 10 ? "" : "\(Int(abs(d.x))) px \(d.x > 0 ? "right" : "left")"
        let v = abs(d.y) < 10 ? "" : "\(Int(abs(d.y))) px \(d.y > 0 ? "down" : "up")"
        let parts = [h, v].filter { !$0.isEmpty }
        return parts.isEmpty ? "right on top of the wizard" : parts.joined(separator: " and ")
    }

    /// Distance the wizard could walk in direction `v` before hitting a wall or obstacle.
    private func freeDistance(_ v: V2) -> Int {
        let n = simd_normalize(v)
        var d: Float = 0
        while d < 400 && isFree(wizard + n * (d + 8)) { d += 8 }
        return Int(d)
    }

    private func describeState() -> String {
        let w = Game.world
        var lines: [String] = []
        if isRoom {
            lines.append("A young wizard walks around a dungeon room (\(spec.name)). Open floor the wizard can walk " +
                         "before hitting a wall, pit or obstacle: " +
                         Game.dirs.map { "\($0.key.replacingOccurrences(of: "_", with: " ")) \(freeDistance($0.vec)) px" }
                            .joined(separator: ", ") + ".")
        } else {
            lines.append("A young wizard walks around a \(Int(w.x)) by \(Int(w.y)) px meadow. It is \(Int(wizard.x)) px from the left wall, " +
                         "\(Int(w.x - wizard.x)) px from the right wall, \(Int(wizard.y)) px from the top wall and " +
                         "\(Int(w.y - wizard.y)) px from the bottom wall.")
        }
        if let s = penguin {
            lines.append("The goal penguin is \(rel(s)) of the wizard (distance \(Int(simd_distance(s, wizard))) px).")
        } else {
            lines.append("The goal penguin has not appeared yet.")
        }
        let near = ghosts.sorted { simd_distance($0.p, wizard) < simd_distance($1.p, wizard) }.prefix(6)
        lines.append("Dangerous ghosts (touching one means losing), nearest first:")
        for g in near {
            let toward = simd_dot(g.v, wizard - g.p) > 0 ? "drifting toward the wizard" : "drifting away from the wizard"
            let armed = g.age >= ghostArm ? "" : ", still harmless for a moment"
            lines.append("- \(rel(g.p)) (distance \(Int(simd_distance(g.p, wizard))) px), \(toward) at " +
                         "\(Int(simd_length(g.v))) px/s\(armed)")
        }
        return lines.joined(separator: "\n")
    }

    private func requestMove() {
        pending = true
        aiThinking = true
        let myEpoch = epoch
        let model = player
        let safe = isRoom ? 100 : 150
        let blockedNote = " If a direction is blocked by a wall or obstacle, the wizard slides along it and may not " +
            "move at all that tick, so prefer directions with open floor in the distances given."
        let instructions: String = penguin != nil
            ? "Choose the direction the wizard should walk for the next third of a second. It must get to the penguin " +
              "while staying clear of every ghost, especially ones drifting toward it, and it cannot walk through " +
              "walls or obstacles. If a ghost is within \(safe) px, prefer safety over speed." + blockedNote
            : "Choose the direction the wizard should walk for the next third of a second. The penguin hasn't " +
              "appeared, so move to the safest spot: away from every ghost and away from walls and corners." + blockedNote
        let options = Dictionary(uniqueKeysWithValues: Game.dirs.map { ($0.key, $0.desc) })
        let state = describeState()

        Task { @MainActor [weak self, client] in
            do {
                let d = try await client.decide(model: model, state: state,
                                                instructions: instructions, options: options)
                guard let self, self.epoch == myEpoch else { return }
                self.aiDir = Game.dirs.first { $0.key == d.choice }?.vec ?? .zero
                self.moveLeft = self.moveTime
                self.pending = false
                self.aiThinking = false
                self.aiMoves += 1
                if ProcessInfo.processInfo.environment["SF_PLAYER"] != nil {
                    NSLog("AI move \(self.aiMoves): \(d.choice) conf=\(d.confidence) \(Int(d.latency * 1000))ms elapsed=\(self.elapsed) score=\(self.score)")
                }
                self.aiProbs = Game.dirs.map {
                    DirProb(id: $0.key, label: $0.label, p: d.probabilities[$0.key] ?? 0)
                }
                self.aiStatus = String(format: "%@ · %.0f%% sure · %.1fs",
                                       d.choice.replacingOccurrences(of: "_", with: " "),
                                       d.confidence * 100, d.latency)
            } catch {
                guard let self, self.epoch == myEpoch else { return }
                self.pending = false
                self.aiThinking = false
                self.aiError = error.localizedDescription
                self.aiStatus = error.localizedDescription
            }
        }
    }

    // MARK: scene
    private func easeOutBack(_ t: Float) -> Float {
        let x = min(1, max(0, t)), c1: Float = 1.70158, c3 = c1 + 1
        return 1 + c3 * pow(x - 1, 3) + c1 * pow(x - 1, 2)
    }

    /// uv rect of one frame in a `cols` x 8 sheet.
    private func frame(_ col: Int, _ row: Int, cols: Int) -> SIMD4<Float> {
        SIMD4(Float(col) / Float(cols), Float(row) / 8, 1 / Float(cols), 1 / 8)
    }

    /// Light pools for the night shader: wizard lantern, ghost glows, penguin beacon, spawn cracks, scenery.
    func lights() -> [LightRec] {
        var out: [LightRec] = []
        let lanternFade: Float = state == .lost ? max(0, 1 - endClock * 1.1) : 1
        let flicker = 0.94 + 0.06 * sin(clock * 13) * sin(clock * 5.3)
        let hRoom = isRoom
        let scale: Float = hRoom ? 0.62 : 1       // rooms are smaller than the meadow
        let hL: Float = hRoom ? 85 : 40            // light heights above the floor
        out.append(LightRec(a: SIMD4(wizard.x, wizard.y, 430 * scale, 1.05 * flicker * lanternFade),
                            color: SIMD4(1.0, 0.84, 0.55, hL)))
        if let s = penguin {
            let k = easeOutBack((clock - penguinBorn) / 0.4)
            out.append(LightRec(a: SIMD4(s.x, s.y, 270 * scale * k, 0.5 + 0.12 * sin(clock * 4)),
                                color: SIMD4(1.0, 0.86, 0.42, hRoom ? 50 : 30)))
        }
        for g in ghosts {
            let t = min(1, g.age / ghostArm)
            out.append(LightRec(a: SIMD4(g.p.x, g.p.y, 210 * scale, 0.55 * t),
                                color: SIMD4(0.40, 0.62, 1.0, hRoom ? 80 : 40)))
            if g.age < ghostArm + 0.4 && out.count < 24 {
                let fade: Float = g.age < ghostArm ? 1 : max(0, 1 - (g.age - ghostArm) / 0.4)
                out.append(LightRec(a: SIMD4(g.p.x, g.p.y + (hRoom ? 0 : 46), 170 * scale,
                                              0.8 * fade * (0.7 + 0.3 * sin(clock * 16))),
                                    color: SIMD4(1.0, 0.42, 0.12, hRoom ? 8 : 20)))
            }
            if out.count >= 24 { break }
        }
        if hRoom {
            for (i, gl) in levelGlows.enumerated() where out.count < 32 {
                let flick = 0.85 + 0.15 * sin(clock * 15 + Float(i) * 1.9) * sin(clock * 6.1 + Float(i))
                out.append(LightRec(a: SIMD4(gl.a.x, gl.a.y, gl.a.z, gl.a.w * flick), color: gl.color))
            }
        } else {
            for (i, pr) in props.enumerated() {
                guard let gl = pr.glow, out.count < 32, let r = PropAtlas.rects[pr.name] else { continue }
                let height = r.w * pr.scale                // r = (x, y, width, height) in atlas pixels
                let flick = 0.86 + 0.14 * sin(clock * 17 + Float(i) * 1.7) * sin(clock * 7.3 + Float(i))
                out.append(LightRec(a: SIMD4(pr.p.x, pr.p.y - height * gl.lift, gl.radius, gl.intensity * flick),
                                    color: SIMD4(gl.color.x, gl.color.y, gl.color.z, 40)))
            }
        }
        return out
    }

    func buildScene() -> [SceneItem] {
        let w = Game.world
        var items: [SceneItem] = []
        let room = isRoom
        let K = StageCam.depthK
        let tanP = StageCam.sinP / StageCam.cosP
        let billSlope: Float = room ? K * tanP : 0          // standing sprites: nearer the higher they reach
        let decalSlope: Float = room ? -K / tanP : 0        // flat decals: nearer further down the screen

        func add(_ tex: TexID?, _ kind: Kind, _ c: V2, _ size: V2, tint: SIMD4<Float> = SIMD4(1, 1, 1, 1),
                 rot: Float = 0, param: Float = 0, seed: Float = 0, uv: SIMD4<Float> = SIMD4(0, 0, 1, 1),
                 depth: Float = 0, feetY: Float = 0, slope: Float = 0, g: V2 = .zero) {
            items.append(SceneItem(tex: tex, inst: SpriteInstance(
                center: c, size: size, tint: tint, rotation: rot, kind: kind.rawValue, param: param,
                seed: seed, uv: uv, depthInfo: SIMD4(depth, feetY, slope, 0), gpos: SIMD4(g.x, g.y, 0, 0))))
        }

        if !room {
            add(.ground, .ground, w / 2, w + V2(160, 160), depth: 0.99, g: w / 2)      // oversized so camera shake never shows the edge
        }

        // spawn cracks (ground layer)
        for g in ghosts where g.age < ghostArm + 0.4 {
            let t = min(1, g.age / ghostArm)
            let fade: Float = g.age < ghostArm ? 1 : max(0, 1 - (g.age - ghostArm) / 0.4)
            let pulse = 0.65 + 0.35 * sin(clock * 16)
            let s = 190 * rs * (1.15 - 0.15 * t)
            let at = place(g.p, feetOff: 46 * rs).feet
            add(.cracks, .emissive, room ? at : g.p + V2(0, 46), V2(s, s * (room ? StageCam.sinP : 1)),
                tint: SIMD4(1, 1, 1, 0.95 * fade * pulse),
                depth: floorDepth(g.p) - 0.0015, feetY: room ? at.y : 0, slope: decalSlope, g: g.p)
        }

        // shadows (on the floor)
        func shadow(_ g: V2, _ width: Float, feet: Float, _ a: Float = 0.5) {
            let f = place(g, feetOff: feet).feet
            add(nil, .shadow, f, V2(width, width * 0.36), tint: SIMD4(0.02, 0.07, 0.02, a),
                depth: floorDepth(g) - 0.001, feetY: f.y, slope: decalSlope, g: g)
        }
        for g in ghosts { shadow(g.p, 110 * rs * easeOutBack(g.age / 0.5), feet: 58 * rs, 0.28) }
        if let s = penguin { shadow(s, 130 * rs * easeOutBack((clock - penguinBorn) / 0.4), feet: 76 * rs) }
        let winHop: Float = state == .won ? max(0.5, 1 - 0.12 * abs(sin(endClock * 7))) : 1
        shadow(wizard, 100 * rs * winHop, feet: 56 * rs)

        // characters + scenery, back to front
        struct Ent { var y: Float; var item: SceneItem }
        var ents: [Ent] = []
        func ent(_ tex: TexID, sortY: Float, center: V2, feet: V2, g: V2, _ size: V2, rot: Float = 0,
                 param: Float = 0, tint: SIMD4<Float> = SIMD4(1, 1, 1, 1), seed: Float = 0,
                 uv: SIMD4<Float> = SIMD4(0, 0, 1, 1), kind: Kind = .sprite) {
            let inst = SpriteInstance(center: center, size: size, tint: tint, rotation: rot,
                                      kind: kind.rawValue, param: param, seed: seed, uv: uv,
                                      depthInfo: SIMD4(room ? floorDepth(g) - 0.004 : 0.5, feet.y, billSlope, 0),
                                      gpos: SIMD4(g.x, g.y, 0, 0))
            ents.append(Ent(y: sortY, item: SceneItem(tex: tex, inst: inst)))
        }

        if !room {
            for pr in props {
                guard let r = PropAtlas.rects[pr.name] else { continue }
                let size = V2(r.z, r.w) * pr.scale
                let uv = SIMD4<Float>(r.x / PropAtlas.size.x, r.y / PropAtlas.size.y, r.z / PropAtlas.size.x, r.w / PropAtlas.size.y)
                let c = pr.p - V2(0, size.y / 2)
                ent(.props, sortY: pr.flat ? -1000 + pr.p.y : pr.p.y, center: c, feet: pr.p, g: c, size,
                    tint: SIMD4(1, 1, 1, pr.alpha), uv: uv)
            }
        }

        for g in ghosts {
            let k = easeOutBack(g.age / 0.5)
            let armed = g.age >= ghostArm
            let alpha: Float = armed ? 1 : 0.25 + 0.6 * (g.age / ghostArm)
            let bob = 5 * rs * sin(clock * 3 + g.phase)
            let col = Int(clock * 8 + g.phase * 2) % 6
            let row = dirRow(g.v)
            let size = spriteSize * k
            let pl = place(g.p, feetOff: 58 * rs)
            let flicker = 0.9 + 0.1 * sin(clock * 9 + g.phase * 3)
            ent(.ghostWalk, sortY: g.p.y, center: pl.center + V2(0, bob - 4 * rs), feet: pl.feet, g: g.p,
                V2(size, size), tint: SIMD4(0.78, 0.9, 1.0, alpha * flicker),
                uv: frame(col, row, cols: 6), kind: .emissive)
        }

        if let s = penguin {
            let born = clock - penguinBorn
            let k = easeOutBack(born / 0.4)
            let phase = clock.truncatingRemainder(dividingBy: 1.3)
            let hop = (phase < 0.45 ? sin(phase / 0.45 * .pi) * 22 : 0) * rs
            let squash: Float = phase < 0.45 ? 1 + 0.06 * sin(phase / 0.45 * .pi)
                                              : 1 - 0.04 * exp(-(phase - 0.45) * 12)
            // The penguin watches the wizard. Its sheet has no up-left row, so mirror up-right.
            var row = dirRow(wizard - s)
            var mirror: Float = 1
            if row == 3 { row = 5; mirror = -1 }
            let col = Int(clock * 3) % 4
            let size: Float = 180 * rs * k
            let pl = place(s, feetOff: 76 * rs)
            ent(.penguin, sortY: s.y, center: pl.center + V2(0, -7 * rs - hop), feet: pl.feet, g: s,
                V2(size * mirror / squash, size * squash), seed: 1, uv: frame(col, row, cols: 4))
        }

        do {
            let pl = place(wizard, feetOff: 57 * rs)
            var c = pl.center
            var size = V2(spriteSize, spriteSize)
            var rot: Float = 0
            var tint = SIMD4<Float>(1, 1, 1, 1)
            var tex = TexID.wizardIdle
            var uv = frame(Int(clock * 4) % 4, faceRow, cols: 4)
            switch state {
            case .playing:
                if moving {
                    tex = .wizardWalk
                    uv = frame(Int(walkPhase) % 6, faceRow, cols: 6)
                }
            case .won:
                tex = .wizardIdle
                uv = frame(Int(clock * 6) % 4, 0, cols: 4)
                c.y -= abs(sin(endClock * 7)) * 46 * rs
            case .lost:
                tex = .wizardWalk
                uv = frame(0, faceRow, cols: 6)
                rot = min(endClock * 4, 1) * 1.45
                size *= max(0.75, 1 - endClock * 0.2)
                tint = SIMD4(1, 0.6, 0.6, max(0.2, 1 - endClock * 0.5))
            }
            ent(tex, sortY: wizard.y, center: c, feet: pl.feet, g: wizard, size, rot: rot, tint: tint, uv: uv)
        }
        for e in ents.sorted(by: { $0.y < $1.y }) { items.append(e.item) }

        // particles (plane space, always in front)
        for p in particles {
            let a = min(1, p.life / p.maxLife * 1.6) * 0.7
            add(nil, .glow, p.p, V2(p.size, p.size) * 2, tint: SIMD4(p.color.x, p.color.y, p.color.z, a), depth: 0.0)
        }

        add(nil, .vignette, w / 2, w, depth: 0.0)
        return items
    }
}
