import SwiftUI

@main
struct SquirrelFinderApp: App {
    @StateObject private var game = Game()

    init() {
        NSApplication.shared.setActivationPolicy(.regular)
    }

    var body: some Scene {
        WindowGroup("Squirrel Finder") {
            ContentView(game: game)
                .frame(minWidth: 960, minHeight: 600)
                .onAppear {
                    NSApp.activate(ignoringOtherApps: true)
                    HUDSnapshot.runIfRequested(game)
                }
        }
        .defaultSize(width: 1280, height: 800)
        .windowResizability(.contentMinSize)
    }
}

struct ContentView: View {
    @ObservedObject var game: Game

    var body: some View {
        ZStack {
            MetalView(game: game)
                .ignoresSafeArea()
            HUD(game: game)
                .padding(18)
        }
        .background(Color.black)
    }
}

/// Number whose changed digits slide in from the left while the old digit leaves to the right.
struct DigitTicker: View {
    let value: Int
    var size: CGFloat = 24

    var body: some View {
        let digits = Array(String(value))
        HStack(spacing: 0) {
            ForEach(0..<digits.count, id: \.self) { i in
                ZStack {
                    Text(String(digits[i]))
                        .id(digits[i])
                        .transition(.asymmetric(
                            insertion: .move(edge: .leading).combined(with: .opacity),
                            removal: .move(edge: .trailing).combined(with: .opacity)))
                }
                .frame(width: size * 0.62, height: size * 1.25)
                .clipped()
            }
        }
        .font(.system(size: size, weight: .bold, design: .rounded))
        .monospacedDigit()
        .animation(.easeOut(duration: 0.16), value: value)
    }
}

struct HUD: View {
    @ObservedObject var game: Game

    private func clockString(_ t: Double) -> String {
        let m = Int(t) / 60
        return String(format: "%d:%04.1f", m, t - Double(m * 60))
    }

    private func chip<Content: View>(_ title: String, symbol: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Label(title, systemImage: symbol)
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .opacity(0.65)
            content()
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                chip("TIME", symbol: "stopwatch") {
                    Text(clockString(game.shownTime))
                        .font(.system(size: 24, weight: .bold, design: .rounded))
                        .monospacedDigit()
                        .contentTransition(.numericText(countsDown: false))
                        .animation(.snappy(duration: 0.18), value: game.shownTime)
                }
                chip("LEVEL \(game.levelIndex + 1)/\(game.levelCount)", symbol: "flag.fill") {
                    VStack(alignment: .leading, spacing: 0) {
                        Text(game.levelName)
                            .font(.system(size: 15, weight: .bold, design: .rounded))
                        Text(game.levelTagline)
                            .font(.system(size: 10, weight: .medium, design: .rounded)).opacity(0.7)
                    }
                    .frame(minWidth: 130, alignment: .leading)
                }
                chip("SCORE", symbol: "star.fill") { DigitTicker(value: game.score) }
                chip("BEST", symbol: "trophy.fill") {
                    Text("\(game.bestScore)")
                        .font(.system(size: 24, weight: .bold, design: .rounded))
                        .monospacedDigit()
                        .opacity(0.8)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    Picker("Player", selection: $game.player) {
                        Text("🧙 You").tag("")
                        ForEach(game.models, id: \.self) { Text("🤖 \($0)").tag($0) }
                    }
                    .pickerStyle(.menu)
                    .fixedSize()
                    if !game.aiStatus.isEmpty {
                        HStack(spacing: 6) {
                            if game.aiThinking { ProgressView().controlSize(.small) }
                            Text(game.aiThinking ? "thinking…" : game.aiStatus)
                                .font(.system(size: 12, weight: .medium, design: .rounded))
                                .foregroundStyle(game.aiError == nil ? Color.primary : Color.red)
                        }
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                    }
                }
            }

            if !game.player.isEmpty && !game.aiProbs.isEmpty {
                HStack(spacing: 6) {
                    ForEach(game.aiProbs) { d in
                        VStack(spacing: 3) {
                            ZStack(alignment: .bottom) {
                                RoundedRectangle(cornerRadius: 3).fill(.white.opacity(0.15)).frame(width: 14, height: 34)
                                RoundedRectangle(cornerRadius: 3).fill(.white)
                                    .frame(width: 14, height: max(2, 34 * d.p))
                            }
                            Text(d.label).font(.system(size: 11, weight: .semibold))
                        }
                    }
                }
                .padding(8)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            }

            Spacer()

            if game.state != .playing {
                HStack {
                    Spacer()
                    VStack(spacing: 12) {
                        Text(game.state == .won ? "You found the penguin! 🐧" : "Caught by a ghost 👻")
                            .font(.system(size: 34, weight: .heavy, design: .rounded))
                        Text("Score \(game.score)")
                            .font(.system(size: 26, weight: .bold, design: .rounded))
                        if game.state == .won {
                            Text("includes +\(game.bonus) for the catch")
                                .font(.system(size: 14, weight: .medium, design: .rounded)).opacity(0.75)
                        }
                        Text(clockString(game.shownTime))
                            .font(.system(size: 16, weight: .medium, design: .rounded)).opacity(0.8)
                        if game.state == .won {
                            Button(game.isLastLevel ? "Start over, harder" : "Next level ▶") { game.nextLevel() }
                                .keyboardShortcut(.defaultAction)
                                .controlSize(.large)
                        } else {
                            Button("Try again") { game.reset() }
                                .keyboardShortcut(.defaultAction)
                                .controlSize(.large)
                        }
                        Text("or press Space").font(.caption).opacity(0.6)
                    }
                    .padding(28)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
                    Spacer()
                }
                Spacer()
            }

            if game.state == .playing && game.player.isEmpty {
                Text("Arrow keys / WASD to move · dodge 👻 · catch the 🐧")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
            }
        }
    }
}

/// Debug aid: SF_HUD_SNAPSHOT=/path/out.png renders the HUD (with sample numbers) over green and quits.
enum HUDSnapshot {
    @MainActor static func runIfRequested(_ game: Game) {
        guard let out = ProcessInfo.processInfo.environment["SF_HUD_SNAPSHOT"] else { return }
        game.score = 1234
        game.shownTime = 75.3
        game.bestScore = 2210
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            let view = HUD(game: game).padding(18).frame(width: 1280, height: 400).background(Color(red: 0.2, green: 0.5, blue: 0.2))
            let r = ImageRenderer(content: view)
            r.scale = 1
            if let img = r.nsImage, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
               let png = rep.representation(using: .png, properties: [:]) {
                try? png.write(to: URL(fileURLWithPath: out))
            }
            exit(0)
        }
    }
}
