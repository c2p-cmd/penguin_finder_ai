import Foundation

/// One answer from an Ollama decision model.
struct Decision: Sendable {
    let choice: String
    let probabilities: [String: Double]
    let confidence: Double
    let latency: TimeInterval
}

enum OllamaError: LocalizedError {
    case unreachable
    case server(String)
    case malformed

    var errorDescription: String? {
        switch self {
        case .unreachable: return "Ollama isn't reachable on 127.0.0.1:11434"
        case .server(let m): return m
        case .malformed: return "Unexpected reply from Ollama"
        }
    }
}

/// Talks to the local Ollama server. The installed models only have the "decision"
/// capability, so chat/generate are rejected; they are driven through /v1/systemone.
struct OllamaClient: Sendable {
    static let base = URL(string: "http://127.0.0.1:11434")!

    private func send(_ req: URLRequest) async throws -> [String: Any] {
        let data: Data
        let resp: URLResponse
        do {
            (data, resp) = try await URLSession.shared.data(for: req)
        } catch {
            throw OllamaError.unreachable
        }
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw OllamaError.malformed
        }
        if let http = resp as? HTTPURLResponse, http.statusCode >= 400 {
            throw OllamaError.server((obj["error"] as? String) ?? "HTTP \(http.statusCode)")
        }
        return obj
    }

    func models() async throws -> [String] {
        var req = URLRequest(url: Self.base.appendingPathComponent("v1/models"))
        req.timeoutInterval = 5
        let obj = try await send(req)
        let list = obj["data"] as? [[String: Any]] ?? []
        return list.compactMap { $0["id"] as? String }
    }

    /// Ask `model` to pick one of `options` (key -> description) given `state`.
    func decide(model: String, state: String, instructions: String,
                options: [String: String]) async throws -> Decision {
        var req = URLRequest(url: Self.base.appendingPathComponent("v1/systemone"))
        req.httpMethod = "POST"
        req.timeoutInterval = 90
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "model": model,
            "state": state,
            "questions": [
                "move": [
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": options,
                ]
            ],
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)

        let t0 = Date()
        let obj = try await send(req)
        guard let answers = obj["answers"] as? [String: Any],
              let move = answers["move"] as? [String: Any],
              let choice = move["choice"] as? String else { throw OllamaError.malformed }
        let probs = (move["probabilities"] as? [String: Double]) ?? [:]
        return Decision(choice: choice, probabilities: probs,
                        confidence: (move["confidence"] as? Double) ?? 0,
                        latency: Date().timeIntervalSince(t0))
    }
}
