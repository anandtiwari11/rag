#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "fuzzy_name_search.hpp"
#include "map_loader.hpp"

// One method with how well it matched the query.
struct RankedMethod {
    const MethodInfo* method = nullptr;
    double score = 0.0;
    double fuzzy = 0.0;     // whole-name Levenshtein similarity (first tie-breaker)
    double coverage = 0.0;  // fraction of the method-name tokens hit by keywords (second tie-breaker)
    std::string reason;     // "exact" | "fuzzy" | "token" | "caller"
};

namespace rank_detail {

inline std::vector<std::string> split_snake(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == '_') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Very light stemming so "created" ~ "create", "invoices" ~ "invoice".
inline std::string stem(const std::string& w) {
    auto ends_with = [&](const char* suf) {
        std::string s(suf);
        return w.size() > s.size() + 2 && w.compare(w.size() - s.size(), s.size(), s) == 0;
    };
    if (ends_with("ing")) return w.substr(0, w.size() - 3);
    if (ends_with("ed")) return w.substr(0, w.size() - 2);
    if (ends_with("es")) return w.substr(0, w.size() - 2);
    if (ends_with("s")) return w.substr(0, w.size() - 1);
    return w;
}

// How well does ONE keyword match ONE name token?
inline double keyword_vs_token(const std::string& kw, const std::string& tok) {
    if (kw == tok) return 100.0;
    const std::string ks = stem(kw), ts = stem(tok);
    if (ks == ts) return 95.0;
    // Stems that only differ by a trailing letter or two: created/create → creat/create.
    if (ks.size() >= 4 && ts.size() >= 4 &&
        (ts.compare(0, ks.size(), ks) == 0 || ks.compare(0, ts.size(), ts) == 0) &&
        std::max(ks.size(), ts.size()) - std::min(ks.size(), ts.size()) <= 2) {
        return 95.0;
    }
    if (kw.size() >= 3 && tok.compare(0, kw.size(), kw) == 0) return 85.0;   // prefix
    if (ks.size() >= 3 && ts.compare(0, ks.size(), ks) == 0) return 82.0;
    if (kw.size() >= 4 && tok.find(kw) != std::string::npos) return 75.0;   // substring
    // Typo inside the token: reconciliatio vs reconciliation → Levenshtein.
    if (kw.size() >= 4 && tok.size() >= 4) {
        return FuzzyNameSearch::fuzzy_score(kw, tok) * 0.9;
    }
    return 0.0;
}

// Score a method name against the split keywords of the query.
//
//   score = 0.85 * (average best match per keyword)
//         + 15   * (fraction of the name's tokens explained by the query)
//
// Averaging means EVERY keyword has to match for a high score
// ("invoice created" → create_invoice ≈ 98, invoice_receiver ≈ 60).
// Coverage prefers names that are mostly about the query
// (create_invoice beats test_invoice_entry_created).
inline double token_score(const std::vector<std::string>& keywords,
                          const std::string& name_norm,
                          double* coverage_out) {
    const auto tokens = split_snake(name_norm);
    if (keywords.empty() || tokens.empty()) {
        if (coverage_out) *coverage_out = 0.0;
        return 0.0;
    }

    double sum = 0.0;
    std::vector<bool> token_hit(tokens.size(), false);

    for (const auto& kw : keywords) {
        double kw_best = 0.0;
        for (size_t t = 0; t < tokens.size(); ++t) {
            double s = keyword_vs_token(kw, tokens[t]);
            if (s > kw_best) kw_best = s;
            if (s >= 70.0) token_hit[t] = true;
        }
        if (kw.size() >= 4 && name_norm.find(kw) != std::string::npos) kw_best = std::max(kw_best, 80.0);
        sum += kw_best;
    }
    const double mean = sum / keywords.size();

    int covered = 0;
    for (bool h : token_hit) covered += h ? 1 : 0;
    const double coverage = static_cast<double>(covered) / tokens.size();

    if (coverage_out) *coverage_out = coverage;
    return mean * 0.85 + coverage * 15.0;
}

// Identifier-looking words the user typed (contain '_'), e.g. run_reconciliation_for_customer.
inline std::vector<std::string> identifiers_in(const std::string& query) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
        if (cur.find('_') != std::string::npos) out.push_back(FuzzyNameSearch::normalize_name(cur));
        cur.clear();
    };
    for (char c : query) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_') cur.push_back(c);
        else flush();
    }
    flush();
    return out;
}

}  // namespace rank_detail

inline bool is_test_code(const MethodInfo& m) {
    return m.file.find("test") != std::string::npos || m.name.compare(0, 5, "test_") == 0;
}

// Rank every method in the map against the query.
//   raw_query : what the user typed
//   keywords  : the query after excluded words were removed
// Uses FuzzyNameSearch (Levenshtein) for whole-name typos and token
// matching for natural-language questions. Returns best → worst.
inline std::vector<RankedMethod> rank_methods(const MethodMap& map,
                                              const std::string& raw_query,
                                              const std::vector<std::string>& keywords,
                                              double min_score) {
    std::string joined;
    for (size_t i = 0; i < keywords.size(); ++i) {
        if (i) joined.push_back('_');
        joined += keywords[i];
    }
    std::string raw_norm;
    for (char c : raw_query) {
        if (c == ' ') raw_norm.push_back('_');
        else raw_norm.push_back(c);
    }
    raw_norm = FuzzyNameSearch::normalize_name(raw_norm);
    const auto identifiers = rank_detail::identifiers_in(raw_query);

    std::vector<RankedMethod> ranked;
    for (const auto& m : map.methods) {
        const std::string n = FuzzyNameSearch::normalize_name(m.name);
        if (n.empty()) continue;

        RankedMethod r;
        r.method = &m;

        bool exact = (n == raw_norm) || (!joined.empty() && n == joined);
        for (const auto& id : identifiers) exact = exact || (n == id);

        if (exact) {
            r.score = 100.0;
            r.fuzzy = 100.0;
            r.coverage = 1.0;
            r.reason = "exact";
        } else {
            // 1) whole-name typo match (your Levenshtein): against the whole
            //    query, the cleaned query, and any identifier typed inside it.
            r.fuzzy = std::max(FuzzyNameSearch::fuzzy_score(raw_norm, n),
                               joined.empty() ? 0.0 : FuzzyNameSearch::fuzzy_score(joined, n));
            for (const auto& id : identifiers) r.fuzzy = std::max(r.fuzzy, FuzzyNameSearch::fuzzy_score(id, n));
            // 2) keyword ↔ token match
            double cov = 0.0;
            double tok = rank_detail::token_score(keywords, n, &cov);

            if (r.fuzzy >= tok) {
                r.score = r.fuzzy;
                r.reason = "fuzzy";
            } else {
                r.score = tok;
                r.reason = "token";
            }
            r.coverage = cov;
        }

        // Questions are usually about production code, not its tests.
        if (r.reason != "exact" && is_test_code(m)) r.score *= 0.92;

        if (r.score >= min_score) ranked.push_back(r);
    }

    // Same score → prefer the name mostly explained by the query, then the
    // name closest to what was typed, then the shorter name.
    std::sort(ranked.begin(), ranked.end(), [](const RankedMethod& a, const RankedMethod& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.coverage != b.coverage) return a.coverage > b.coverage;
        if (a.fuzzy != b.fuzzy) return a.fuzzy > b.fuzzy;
        if (a.method->name.size() != b.method->name.size()) return a.method->name.size() < b.method->name.size();
        return a.method->name < b.method->name;
    });
    return ranked;
}

// Natural-language questions ("how is invoice created") are not a method name.
// Keep methods that share the most specific word (invoice, not every create_*),
// then sort so names that also match the other words come first.
inline std::vector<RankedMethod> rank_topic_methods(const MethodMap& map,
                                                    const std::vector<std::string>& keywords) {
    std::vector<std::string> content;
    for (const auto& kw : keywords) {
        if (kw.size() >= 4) content.push_back(kw);
    }
    if (content.empty()) content = keywords;
    if (content.empty()) return {};

    auto best_hit = [](const std::string& name, const std::string& kw) {
        double best = 0.0;
        for (const auto& tok : rank_detail::split_snake(name)) {
            best = std::max(best, rank_detail::keyword_vs_token(kw, tok));
        }
        if (kw.size() >= 4 && name.find(kw) != std::string::npos) best = std::max(best, 80.0);
        return best;
    };

    std::vector<std::string> names;
    names.reserve(map.methods.size());
    for (const auto& m : map.methods) names.push_back(FuzzyNameSearch::normalize_name(m.name));

    std::vector<int> hit_count(content.size(), 0);
    std::vector<std::vector<double>> hits(map.methods.size(), std::vector<double>(content.size(), 0.0));
    for (size_t i = 0; i < map.methods.size(); ++i) {
        if (names[i].empty()) continue;
        for (size_t k = 0; k < content.size(); ++k) {
            const double s = best_hit(names[i], content[k]);
            if (s >= 70.0) {
                hits[i][k] = s;
                ++hit_count[k];
            }
        }
    }

    size_t anchor = 0;
    int anchor_hits = 0;
    for (size_t k = 0; k < content.size(); ++k) {
        if (hit_count[k] == 0) continue;
        if (anchor_hits == 0 || hit_count[k] < anchor_hits) {
            anchor = k;
            anchor_hits = hit_count[k];
        }
    }
    if (anchor_hits == 0) return {};

    std::vector<RankedMethod> ranked;
    for (size_t i = 0; i < map.methods.size(); ++i) {
        if (hits[i][anchor] < 70.0) continue;
        int matched = 0;
        double sum = 0.0;
        for (size_t k = 0; k < content.size(); ++k) {
            if (hits[i][k] >= 70.0) {
                ++matched;
                sum += hits[i][k];
            }
        }
        double coverage = 0.0;
        const double tok = rank_detail::token_score(keywords, names[i], &coverage);
        RankedMethod r;
        r.method = &map.methods[i];
        r.fuzzy = 0.0;
        r.coverage = static_cast<double>(matched) / static_cast<double>(content.size());
        r.score = (sum / matched) * 0.5 + matched * 20.0 + tok * 0.25;
        r.reason = "topic";
        if (is_test_code(map.methods[i])) r.score *= 0.92;
        ranked.push_back(r);
    }

    std::sort(ranked.begin(), ranked.end(), [](const RankedMethod& a, const RankedMethod& b) {
        if (a.score != b.score) return a.score > b.score;
        if (a.coverage != b.coverage) return a.coverage > b.coverage;
        if (a.method->name.size() != b.method->name.size()) return a.method->name.size() < b.method->name.size();
        return a.method->name < b.method->name;
    });
    return ranked;
}
