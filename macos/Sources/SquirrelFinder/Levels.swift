import simd

/// One playable stage. `room` is an Assets/rooms/<name>.obj voxel room; nil = the flat graveyard.
struct LevelSpec {
    let name: String
    let tagline: String
    let room: String?
    let ghosts: Int                       // at the start
    let maxGhosts: Int
    let speed: ClosedRange<Float>         // ghost speed, in flat-graveyard units per second
    let spawnEvery: Float                 // base seconds between new ghosts
    let ambient: SIMD3<Float>             // moonlight tint (linear)
}

enum Levels {
    static let all: [LevelSpec] = [
        LevelSpec(name: "Moonlit Graveyard", tagline: "Warm up among the graves", room: nil,
                  ghosts: 4, maxGhosts: 12, speed: 80...170, spawnEvery: 2.6,
                  ambient: SIMD3(0.075, 0.105, 0.20)),
        LevelSpec(name: "The Overgrown Garden", tagline: "Vines, a fountain and a hole in the hedge", room: "garden",
                  ghosts: 4, maxGhosts: 10, speed: 80...160, spawnEvery: 2.8,
                  ambient: SIMD3(0.06, 0.10, 0.15)),
        LevelSpec(name: "Dungeon Stairs", tagline: "Torchlight on the old steps", room: "stairs",
                  ghosts: 5, maxGhosts: 11, speed: 90...170, spawnEvery: 2.6,
                  ambient: SIMD3(0.07, 0.07, 0.16)),
        LevelSpec(name: "Boss Room", tagline: "Something big used to live here", room: "boss",
                  ghosts: 6, maxGhosts: 12, speed: 100...180, spawnEvery: 2.4,
                  ambient: SIMD3(0.09, 0.07, 0.15)),
        LevelSpec(name: "Trapped Room", tagline: "The floor is lava. Mostly.", room: "trapped",
                  ghosts: 6, maxGhosts: 13, speed: 100...190, spawnEvery: 2.2,
                  ambient: SIMD3(0.10, 0.06, 0.12)),
        LevelSpec(name: "King's Throne", tagline: "The court is still in session", room: "throne",
                  ghosts: 7, maxGhosts: 14, speed: 110...200, spawnEvery: 2.0,
                  ambient: SIMD3(0.10, 0.06, 0.13)),
        LevelSpec(name: "Dragon Room", tagline: "Last stop. Mind the river.", room: "dragon",
                  ghosts: 8, maxGhosts: 15, speed: 120...215, spawnEvery: 1.8,
                  ambient: SIMD3(0.06, 0.08, 0.17)),
    ]
}
