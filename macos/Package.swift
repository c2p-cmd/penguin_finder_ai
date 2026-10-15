// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SquirrelFinder",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "SquirrelFinder",
            path: "Sources/SquirrelFinder"
        )
    ]
)
