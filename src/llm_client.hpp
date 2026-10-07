#pragma once

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "json.hpp"

struct ChatMessage {
    std::string role;     // "system" | "user" | "assistant"
    std::string content;
};

// Talks to any OpenAI-compatible local server (Ollama, llama.cpp, vLLM, ...).
// Uses the curl CLI so no extra C++ dependencies are needed.
//
// Environment:
//   LLM_BASE_URL         default http://localhost:11434/v1
//   LLM_MODEL            required, e.g. qwen2.5:3b
//   LLM_TEMPERATURE      default 0.1
//   LLM_MAX_TOKENS       default 700
//   LLM_TIMEOUT_SECONDS  default 180
class LlmClient {
public:
    LlmClient() {
        base_url_ = env_or("LLM_BASE_URL", "http://localhost:11434/v1");
        model_ = env_or("LLM_MODEL", "");
        temperature_ = std::atof(env_or("LLM_TEMPERATURE", "0.1").c_str());
        max_tokens_ = std::atoi(env_or("LLM_MAX_TOKENS", "1000").c_str());
        timeout_seconds_ = std::atoi(env_or("LLM_TIMEOUT_SECONDS", "180").c_str());
        while (!base_url_.empty() && base_url_.back() == '/') base_url_.pop_back();
    }

    bool configured() const { return !model_.empty(); }
    const std::string& model() const { return model_; }
    const std::string& base_url() const { return base_url_; }

    std::string chat(const std::string& system, const std::string& user) const {
        return chat({{"system", system}, {"user", user}});
    }

    std::string chat(const std::vector<ChatMessage>& messages) const {
        if (!configured()) {
            throw std::runtime_error("LLM_MODEL is not set. Example: export LLM_MODEL=qwen2.5:3b");
        }

        nlohmann::json body;
        body["model"] = model_;
        body["temperature"] = temperature_;
        body["max_tokens"] = max_tokens_;
        body["stream"] = false;
        body["messages"] = nlohmann::json::array();
        for (const auto& m : messages) {
            body["messages"].push_back({{"role", m.role}, {"content", m.content}});
        }

        const std::string req_path = "/tmp/rag_llm_req_" + std::to_string(getpid()) + ".json";
        const std::string err_path = "/tmp/rag_llm_err_" + std::to_string(getpid()) + ".txt";
        {
            std::ofstream out(req_path);
            out << body.dump();
        }

        const std::string cmd =
            "curl -sS --max-time " + std::to_string(timeout_seconds_) +
            " -X POST '" + base_url_ + "/chat/completions'" +
            " -H 'Content-Type: application/json'" +
            " -H 'Authorization: Bearer not-needed'" +
            " --data-binary @" + req_path + " 2>" + err_path;

        const std::string response = run_command(cmd);
        std::remove(req_path.c_str());

        if (response.empty()) {
            std::ifstream err(err_path);
            std::ostringstream ess;
            ess << err.rdbuf();
            std::remove(err_path.c_str());
            throw std::runtime_error("LLM request failed (is the server running at " + base_url_ +
                                     "?). curl: " + ess.str());
        }
        std::remove(err_path.c_str());

        auto j = nlohmann::json::parse(response, nullptr, false);
        if (j.is_discarded()) {
            throw std::runtime_error("LLM returned non-JSON: " + response.substr(0, 400));
        }
        if (j.contains("error")) {
            throw std::runtime_error("LLM error: " + j["error"].dump());
        }
        try {
            std::string content = j.at("choices").at(0).at("message").at("content").get<std::string>();
            return trim(strip_think(content));
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Unexpected LLM response: ") + e.what() +
                                     " body=" + response.substr(0, 400));
        }
    }

    // Qwen3 can emit <think>...</think> before the answer; remove it.
    static std::string strip_think(std::string s) {
        for (;;) {
            size_t start = s.find("<think>");
            if (start == std::string::npos) break;
            size_t end = s.find("</think>", start);
            if (end == std::string::npos) {
                s.erase(start);
                break;
            }
            s.erase(start, end + 8 - start);
        }
        return s;
    }

    static std::string trim(const std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    }

private:
    std::string base_url_;
    std::string model_;
    double temperature_ = 0.1;
    int max_tokens_ = 700;
    int timeout_seconds_ = 180;

    static std::string env_or(const char* name, const std::string& fallback) {
        const char* v = std::getenv(name);
        return (v && *v) ? std::string(v) : fallback;
    }

    static std::string run_command(const std::string& cmd) {
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) throw std::runtime_error("popen failed");
        std::string result;
        char buf[4096];
        while (fgets(buf, sizeof(buf), pipe)) result += buf;
        pclose(pipe);
        return result;
    }
};
