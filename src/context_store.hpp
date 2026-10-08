#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "llm_client.hpp"

// The whole conversation lives in utils/context.txt as blocks.
// When the file grows past max_tokens the LLM summarises it.
class ContextStore {
public:
    struct Block {
        std::string role;
        std::string text;
    };

    explicit ContextStore(std::string path, int max_tokens = 1000000)
        : path_(std::move(path)), max_tokens_(max_tokens) {}

    void append(const std::string& role, const std::string& text) {
        std::ofstream out(path_, std::ios::app);
        out << "\n===== " << role << " =====\n" << text << "\n";
    }

    void clear() {
        std::ofstream out(path_, std::ios::trunc);
    }

    std::string read_all() const {
        std::ifstream in(path_);
        if (!in) return "";
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    int estimate_tokens() const { return static_cast<int>(read_all().size() / 4); }
    int max_tokens() const { return max_tokens_; }

    std::vector<Block> blocks() const {
        std::vector<Block> out;
        std::istringstream in(read_all());
        std::string line;
        Block cur;
        bool have = false;
        while (std::getline(in, line)) {
            if (line.size() > 12 && line.compare(0, 6, "===== ") == 0 &&
                line.compare(line.size() - 6, 6, " =====") == 0) {
                if (have) out.push_back(cur);
                cur = Block{line.substr(6, line.size() - 12), ""};
                have = true;
            } else if (have) {
                cur.text += line;
                cur.text += '\n';
            }
        }
        if (have) out.push_back(cur);
        return out;
    }

    // Skip noisy turns so a small model is not trained to say "not found".
    static bool is_noisy_assistant(const std::string& text) {
        const std::string t = LlmClient::trim(text);
        if (t.rfind("[dry-run]", 0) == 0) return true;
        if (t.rfind("NOT_FOUND", 0) == 0) return true;
        if (t.find("could not find any method") != std::string::npos) return true;
        if (t.find("I could not find") != std::string::npos) return true;
        return false;
    }

    // SUMMARY + last few useful USER/ASSISTANT turns (no dry-runs, no NOT_FOUND).
    std::string history_for_prompt(size_t max_chars) const {
        std::string summary;
        std::vector<std::string> turns;
        for (const auto& b : blocks()) {
            if (b.role.rfind("SUMMARY", 0) == 0) {
                summary = b.text;
            } else if (b.role == "USER") {
                turns.push_back("User: " + LlmClient::trim(b.text));
            } else if (b.role == "ASSISTANT") {
                if (is_noisy_assistant(b.text)) continue;
                turns.push_back("Assistant: " + LlmClient::trim(b.text));
            }
        }

        std::string recent;
        for (auto it = turns.rbegin(); it != turns.rend(); ++it) {
            if (recent.size() + it->size() + 2 > max_chars) break;
            recent = *it + "\n\n" + recent;
        }

        std::string out;
        if (!summary.empty()) out += "Summary of earlier conversation:\n" + LlmClient::trim(summary) + "\n\n";
        if (!recent.empty()) out += "Recent conversation:\n" + recent;
        return out;
    }

    bool maybe_summarize(const LlmClient& llm) {
        const int tokens = estimate_tokens();
        if (tokens < max_tokens_) return false;

        std::cerr << "[context] ~" << tokens << " tokens >= " << max_tokens_ << ", summarising...\n";

        std::string dialogue;
        for (const auto& b : blocks()) {
            if (b.role.rfind("SUMMARY", 0) == 0) dialogue += "Earlier summary:\n" + b.text + "\n";
            else if (b.role == "USER") dialogue += "User: " + b.text + "\n";
            else if (b.role == "ASSISTANT" && !is_noisy_assistant(b.text)) {
                dialogue += "Assistant: " + b.text + "\n";
            } else if (b.role == "RETRIEVAL") dialogue += "Methods looked at:\n" + b.text + "\n";
        }
        if (dialogue.size() > 60000) dialogue = dialogue.substr(dialogue.size() - 60000);

        const std::string system =
            "You compress a long conversation between a developer and a codebase assistant. "
            "Keep every question asked, the method names and file paths that were discussed, "
            "and the conclusions reached. Drop code listings. Be concise.";
        std::string summary;
        try {
            summary = llm.chat(system, "Summarise this conversation:\n\n" + dialogue);
        } catch (const std::exception& e) {
            std::cerr << "[context] summarise failed: " << e.what() << " (keeping file as is)\n";
            return false;
        }

        std::ofstream out(path_, std::ios::trunc);
        out << "===== SUMMARY =====\n" << summary << "\n";
        std::cerr << "[context] compressed to ~" << estimate_tokens() << " tokens\n";
        return true;
    }

private:
    std::string path_;
    int max_tokens_;
};
