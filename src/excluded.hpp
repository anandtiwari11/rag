#pragma once

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// Loads utils/excluded.txt and strips those words from a user query.
class ExcludedWords {
public:
    explicit ExcludedWords(const std::string& path) {
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) {
                line.erase(line.begin());
            }
            while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
                line.pop_back();
            }
            if (line.empty() || line[0] == '#') continue;
            words_.insert(to_lower(line));
        }
    }

    bool contains(const std::string& word) const {
        return words_.find(to_lower(word)) != words_.end();
    }

    // Copy of the query → split into words → drop excluded words.
    // "where is invoice created in our system" → ["invoice", "created"]
    std::vector<std::string> keywords(const std::string& query) const {
        std::vector<std::string> kept;
        std::string word;

        auto flush = [&]() {
            if (!word.empty() && word.size() > 1 && words_.find(word) == words_.end()) {
                kept.push_back(word);
            }
            word.clear();
        };

        for (char c : query) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') {
                word.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            } else {
                flush();
            }
        }
        flush();

        // A snake_case identifier the user typed is replaced by its parts
        // (the whole identifier is matched separately with Levenshtein).
        std::vector<std::string> expanded;
        for (const auto& w : kept) {
            if (w.find('_') == std::string::npos) {
                expanded.push_back(w);
            } else {
                std::string part;
                for (char c : w) {
                    if (c == '_') {
                        if (part.size() > 1 && words_.find(part) == words_.end()) expanded.push_back(part);
                        part.clear();
                    } else {
                        part.push_back(c);
                    }
                }
                if (part.size() > 1 && words_.find(part) == words_.end()) expanded.push_back(part);
            }
        }
        return expanded;
    }

    // Keywords joined with '_' so the whole thing looks like a method name.
    std::string clean(const std::string& query) const {
        auto kws = keywords(query);
        if (kws.empty()) return normalize_alnum(query);
        std::ostringstream oss;
        for (size_t i = 0; i < kws.size(); ++i) {
            if (i) oss << '_';
            oss << kws[i];
        }
        return oss.str();
    }

    static std::string to_lower(std::string s) {
        for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    }

    static std::string normalize_alnum(const std::string& s) {
        std::string out;
        for (char c : s) {
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') out.push_back(c);
            else if (c == ' ') out.push_back('_');
        }
        return out;
    }

private:
    std::unordered_set<std::string> words_;
};
