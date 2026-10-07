#pragma once

#include <algorithm>
#include <string>
#include <vector>

struct FuzzyMatch {
    std::string name;
    double score;
};

class FuzzyNameSearch {
public:
    explicit FuzzyNameSearch(int limit = 3, double min_score = 70.0)
        : limit_(limit), min_score_(min_score) {}

    void set_limit(int limit) { limit_ = limit; }
    void set_min_score(double min_score) { min_score_ = min_score; }

    int limit() const { return limit_; }
    double min_score() const { return min_score_; }

    static int levenshtein(const std::string& a, const std::string& b) {
        const int n = static_cast<int>(a.size());
        const int m = static_cast<int>(b.size());
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1));

        for (int i = 0; i <= n; ++i) dp[i][0] = i;
        for (int j = 0; j <= m; ++j) dp[0][j] = j;

        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= m; ++j) {
                if (a[i - 1] == b[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = 1 + std::min({
                        dp[i - 1][j - 1],  // replace
                        dp[i - 1][j],      // delete
                        dp[i][j - 1],      // insert
                    });
                }
            }
        }
        return dp[n][m];
    }

    static int levenshtein_compact(const std::string& a, const std::string& b) {
        const std::string& s = a.size() >= b.size() ? a : b;
        const std::string& t = a.size() >= b.size() ? b : a;

        std::vector<int> prev(t.size() + 1);
        for (size_t j = 0; j <= t.size(); ++j) prev[j] = static_cast<int>(j);

        for (size_t i = 1; i <= s.size(); ++i) {
            std::vector<int> curr(t.size() + 1);
            curr[0] = static_cast<int>(i);
            for (size_t j = 1; j <= t.size(); ++j) {
                if (s[i - 1] == t[j - 1]) {
                    curr[j] = prev[j - 1];
                } else {
                    curr[j] = 1 + std::min({prev[j - 1], prev[j], curr[j - 1]});
                }
            }
            prev = std::move(curr);
        }
        return prev[t.size()];
    }

    static double distance_to_score(int distance, int max_len) {
        if (max_len <= 0) return distance == 0 ? 100.0 : 0.0;
        double score = (1.0 - static_cast<double>(distance) / max_len) * 100.0;
        return std::max(0.0, score);
    }

    static double fuzzy_score(const std::string& query, const std::string& candidate) {
        if (query.empty() && candidate.empty()) return 100.0;
        if (query.empty() || candidate.empty()) return 0.0;
        int dist = levenshtein(query, candidate);
        int max_len = static_cast<int>(std::max(query.size(), candidate.size()));
        return distance_to_score(dist, max_len);
    }

    static std::string normalize_name(std::string name) {
        std::string out;
        out.reserve(name.size());
        for (char c : name) {
            if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_') {
                out.push_back(c);
            }
        }
        return out;
    }

    std::vector<FuzzyMatch> find(
        const std::string& query,
        const std::vector<std::string>& method_names
    ) const {
        std::string q = normalize_name(query);
        std::vector<FuzzyMatch> matches;

        for (const auto& name : method_names) {
            std::string n = normalize_name(name);
            double score = (q == n) ? 100.0 : fuzzy_score(q, n);
            if (score >= min_score_) {
                matches.push_back({name, score});
            }
        }

        std::sort(matches.begin(), matches.end(), [](const FuzzyMatch& a, const FuzzyMatch& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.name < b.name;
        });

        if (static_cast<int>(matches.size()) > limit_) {
            matches.resize(limit_);
        }
        return matches;
    }

private:
    int limit_;
    double min_score_;
};
