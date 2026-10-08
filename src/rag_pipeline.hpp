#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "context_store.hpp"
#include "excluded.hpp"
#include "llm_client.hpp"
#include "map_loader.hpp"
#include "method_notes.hpp"
#include "rank_methods.hpp"
#include "source_reader.hpp"

struct RagConfig {
    int batch_size = 2;               // default batch for vague questions
    int max_batches = 2;              // fewer retries = less waiting on CPU
    double min_score = 70.0;
    double strong_score = 95.0;       // exact / near-exact → send only top-1, no NOT_FOUND retry
    size_t max_method_chars = 6000;   // hard cap per method (~1.5k tokens)
    int method_head_lines = 80;       // keep start of long methods
    int method_tail_lines = 40;       // keep end of long methods
    size_t max_batch_chars = 10000;   // ~2.5k tokens of source total
    size_t history_chars = 2500;      // keep last Q&A for follow-ups like "elaborate"
    int show_matches = 8;
    // Natural-language questions open this many related methods, best first.
    // Env: RAG_TOPIC_LIMIT (clamped to 1..12)
    int topic_limit = 4;
    bool debug = true;
    bool dry_run = false;
    // When context.txt grows past this many tokens (~4 chars each), it is
    // summarised and the file is rewritten. Env: RAG_CONTEXT_MAX_TOKENS
    int context_max_tokens = 1000000;
    // How many times to call the LLM for one answer.
    // Each later pass revises the previous reply so it matches the user's question.
    // Env: RAG_REFINE_PASSES (clamped to 1..5)
    int refine_passes = 3;
    // Saved answers for methods already explained. Empty disables the cache.
    std::string notes_dir;
};

// Optimised for a small local model (Qwen 3B on CPU):
//   - strong match → 1 method, no NOT_FOUND retry
//   - long methods → head+tail only
//   - history → skip dry-runs / "not found" noise
class RagPipeline {
public:
    RagPipeline(const MethodMap& map, const ExcludedWords& excluded,
                const LlmClient& llm, ContextStore& context, RagConfig cfg)
        : map_(map), excluded_(excluded), llm_(llm), context_(context), cfg_(cfg) {}

    RagConfig& config() { return cfg_; }

    void clear_topic() {
        last_methods_.clear();
        last_topic_name_.clear();
    }

    std::string ask(const std::string& query) {
        context_.append("USER", query);

        std::string answer;
        if (is_usage_question(query)) {
            answer = answer_usage(query);
        } else if (should_use_previous_topic(query)) {
            answer = answer_follow_up(query);
        } else {
            answer = answer_explain(query);
        }

        context_.append("ASSISTANT", answer);
        if (!cfg_.dry_run) context_.maybe_summarize(llm_);
        return answer;
    }

    std::string usage_of(const std::string& method_name) {
        context_.append("USER", "where is " + method_name + " used?");
        std::string answer = answer_usage("where is " + method_name + " used?");
        context_.append("ASSISTANT", answer);
        if (!cfg_.dry_run) context_.maybe_summarize(llm_);
        return answer;
    }

private:
    const MethodMap& map_;
    const ExcludedWords& excluded_;
    const LlmClient& llm_;
    ContextStore& context_;
    RagConfig cfg_;
    std::vector<const MethodInfo*> last_methods_;
    std::string last_topic_name_;  // e.g. run_reconciliation_for_customer

    static bool contains_word(const std::string& text, const char* word) {
        return text.find(word) != std::string::npos;
    }

    bool is_usage_question(const std::string& query) const {
        const std::string q = ExcludedWords::to_lower(query);
        return contains_word(q, " used") || contains_word(q, "usage") || contains_word(q, "who calls") ||
               contains_word(q, "who uses") || contains_word(q, "caller") || contains_word(q, "called from") ||
               contains_word(q, "called by") || contains_word(q, "invoked") || contains_word(q, "references to") ||
               contains_word(q, "where is it called") || contains_word(q, "parent method");
    }

    // "can you elaborate", "explain better", "why?", "and then?" — no new method name.
    bool looks_like_follow_up(const std::string& query) const {
        const std::string q = ExcludedWords::to_lower(query);
        static const char* cues[] = {
            "elaborat", "more detail", "more details", "go deeper", "go deeper",
            "in more detail", "simpler", "simpli", "clearer", "better",
            "continue", "go on", "and then", "what about it", "tell me more",
            "break it down", "step by step", "why does it", "how does it",
            "what does it", "explain it", "explain this", "explain that",
            "same method", "same function", "again", "rephrase", "summarise",
            "summarize", "shorter", "longer", "examples", "example",
        };
        for (const char* c : cues) {
            if (contains_word(q, c)) return true;
        }
        // Short pronoun-heavy questions: "why?", "and this?", "what next?"
        const auto kws = excluded_.keywords(query);
        if (kws.empty() && !last_methods_.empty()) return true;
        if (q.size() < 40 && !last_methods_.empty()) {
            if (contains_word(q, " it") || contains_word(q, "it ") || q == "it" ||
                contains_word(q, "this") || contains_word(q, "that") ||
                contains_word(q, "these") || contains_word(q, "those")) {
                return true;
            }
        }
        return false;
    }

    bool should_use_previous_topic(const std::string& query) const {
        if (last_methods_.empty()) return false;
        if (looks_like_follow_up(query)) return true;
        // Stopwords only left → continue previous topic ("??", "ok and then").
        if (excluded_.keywords(query).empty()) return true;
        return false;
    }

    void remember_methods(const std::vector<const MethodInfo*>& methods) {
        last_methods_ = methods;
        if (!methods.empty() && methods.front()) {
            last_topic_name_ = methods.front()->name;
        }
    }

    void debug(const std::string& line) const {
        if (cfg_.debug) std::cout << "  [debug] " << line << "\n";
    }

    std::string describe(const MethodInfo& m) const {
        std::string s = m.qualified_name.empty() ? m.name : m.qualified_name;
        s += "  (" + m.location() + ", " + std::to_string(m.line_count()) + " lines)";
        return s;
    }

    std::string method_block(const MethodInfo& m, int index, double score) const {
        std::ostringstream b;
        b << "----- method #" << index << " -----\n";
        b << "name: " << m.name << "\n";
        if (!m.parent_class.empty()) b << "class: " << m.parent_class << "\n";
        b << "file: " << m.file << "\n";
        b << "lines: " << m.start_line << "-" << m.end_line << "\n";
        if (score > 0) b << "match score: " << static_cast<int>(score) << "/100\n";
        if (!m.called_by.empty()) {
            b << "called by " << m.called_by.size() << " method(s): ";
            int shown = 0;
            for (int id : m.called_by) {
                if (const MethodInfo* p = map_.find_by_id(id)) {
                    if (shown++) b << ", ";
                    b << p->name;
                    if (shown >= 5) break;
                }
            }
            b << "\n";
        }
        b << "source:\n"
          << read_method_source(map_, m, cfg_.max_method_chars,
                                cfg_.method_head_lines, cfg_.method_tail_lines)
          << "\n";
        return b.str();
    }

    // Strong first hit → only that one method (fast + clear for 3B).
    int batch_limit_for(const std::vector<RankedMethod>& ranked) const {
        if (!ranked.empty() && ranked.front().score >= cfg_.strong_score) return 1;
        return cfg_.batch_size;
    }

    std::vector<size_t> next_batch(const std::vector<RankedMethod>& ranked,
                                   size_t& cursor,
                                   int limit) const {
        std::vector<size_t> picked;
        size_t chars = 0;
        while (cursor < ranked.size() && static_cast<int>(picked.size()) < limit) {
            const MethodInfo& m = *ranked[cursor].method;
            size_t est = static_cast<size_t>(
                std::min(m.line_count(), cfg_.method_head_lines + cfg_.method_tail_lines + 2) * 60);
            if (est > cfg_.max_method_chars) est = cfg_.max_method_chars;
            if (!picked.empty() && chars + est > cfg_.max_batch_chars) break;
            picked.push_back(cursor);
            chars += est;
            ++cursor;
        }
        return picked;
    }

    std::string call_llm(const std::string& system, const std::string& user) const {
        if (cfg_.dry_run) {
            return "[dry-run] prompt built (" + std::to_string(user.size()) + " chars, ~" +
                   std::to_string(user.size() / 4) + " tokens). LLM not called.";
        }
        return llm_.chat({{"system", system}, {"user", user}});
    }

    static bool looks_like_not_found(const std::string& answer) {
        const std::string t = LlmClient::trim(answer);
        return t.rfind("NOT_FOUND", 0) == 0;
    }

    static bool looks_like_enough(const std::string& answer) {
        const std::string t = LlmClient::trim(answer);
        return t == "ENOUGH" || t.rfind("ENOUGH", 0) == 0;
    }

    // Calls the model once, then optionally revises that reply.
    // The user's question decides the kind of answer (summary, business logic,
    // callers, a single line, and so on). Later passes only make the reply
    // match that question more closely. They do not switch the topic.
    // Pass 1 returning NOT_FOUND stops the loop so retrieval can try the next batch.
    MethodNotes notes() const { return MethodNotes(cfg_.notes_dir, map_); }

    // Attach any saved notes for these methods. Returns how many were found.
    int append_saved_notes(std::ostream& out, const std::vector<const MethodInfo*>& methods) const {
        if (cfg_.notes_dir.empty()) return 0;
        int found = 0;
        MethodNotes store = notes();
        for (const MethodInfo* method : methods) {
            if (!method) continue;
            std::string saved = store.load(*method);
            if (saved.empty()) continue;
            if (saved.size() > 4000) saved.resize(4000);
            ++found;
            debug("using saved note " + store.path_for(*method));
            out << "Pre-context from utils/cache for " << method->name << ":\n";
            out << saved << "\n\n";
        }
        return found;
    }

    void remember_answer(const MethodInfo* method, const std::string& answer) const {
        if (!method || cfg_.dry_run || cfg_.notes_dir.empty()) return;
        if (answer.empty() || looks_like_not_found(answer)) return;
        if (answer.rfind("[dry-run]", 0) == 0) return;
        notes().save(*method, answer);
        debug("saved note " + notes().path_for(*method));
    }

    std::string refine_answer(const std::string& question,
                              const std::string& source_block,
                              bool allow_not_found,
                              int pass_limit = -1) const {
        int passes = std::max(1, std::min(cfg_.refine_passes, 5));
        if (pass_limit > 0) passes = std::min(passes, pass_limit);
        if (cfg_.dry_run) {
            debug("refine loop skipped (dry-run); configured passes=" + std::to_string(passes));
            return call_llm("dry-run", question + "\n\n" + source_block);
        }

        const std::string base =
            "You answer questions about code.\n"
            "Use ONLY the source and any pre-context provided. Do not invent.\n"
            "The methods were retrieved because they relate to the question.\n"
            "Answer the user's question directly. Match the kind of answer they asked for.\n"
            "Do not say NOT_FOUND because the question is not the exact function name.\n"
            "If the source was truncated, say the omitted part was not shown.\n";

        std::string notes;
        for (int pass = 1; pass <= passes; ++pass) {
            std::string system = base;
            std::string task;
            if (pass == 1) {
                if (allow_not_found) {
                    system +=
                        "If the source does not contain the answer, reply with exactly NOT_FOUND "
                        "on the first line.\n";
                }
                task = "Answer the question.";
            } else if (pass == passes) {
                system +=
                    "Rewrite the previous answer as the reply the user should see.\n"
                    "Keep the same kind of answer they asked for. Do not add a different structure.\n"
                    "Drop anything the source does not support.\n";
                task = "Write the final reply to the user's question.";
            } else {
                system +=
                    "Revise the previous answer so it follows the user's question more closely.\n"
                    "Stay on what they asked. If it already does that, reply with exactly ENOUGH.\n";
                task = "Revise the previous answer, or reply ENOUGH.";
            }

            std::ostringstream user;
            user << "Question:\n" << question << "\n\n";
            if (!notes.empty()) user << "Previous answer:\n" << notes << "\n\n";
            user << source_block << "\n" << task << "\n";

            debug("llm pass " + std::to_string(pass) + "/" + std::to_string(passes) +
                  " (~" + std::to_string(user.str().size() / 4) + " tokens)");
            const std::string reply = call_llm(system, user.str());

            if (pass == 1 && allow_not_found && looks_like_not_found(reply)) {
                debug("pass 1 said NOT_FOUND; stopping refine loop");
                return reply;
            }
            if (pass > 1 && looks_like_enough(reply)) {
                debug("model said ENOUGH; keeping the previous pass");
                break;
            }
            notes = reply;
        }
        return notes;
    }

    std::string answer_explain(const std::string& query) {
        const auto keywords = excluded_.keywords(query);
        {
            std::string kw;
            for (const auto& k : keywords) kw += k + " ";
            debug("keywords after removing excluded words: " + (kw.empty() ? "(none)" : kw));
        }

        const auto ranked = rank_methods(map_, query, keywords, cfg_.min_score);
        debug("methods above score " + std::to_string(static_cast<int>(cfg_.min_score)) + ": " +
              std::to_string(ranked.size()));

        // "how is invoice created" is not a method name. Open the related methods.
        if (!names_one_method(query, ranked)) {
            return answer_topic(query, keywords);
        }

        std::ostringstream retrieval;
        for (size_t i = 0; i < ranked.size() && static_cast<int>(i) < cfg_.show_matches; ++i) {
            const auto& r = ranked[i];
            std::ostringstream line;
            line << (i + 1) << ". " << static_cast<int>(r.score) << " [" << r.reason << "] "
                 << describe(*r.method);
            debug(line.str());
            retrieval << line.str() << "\n";
        }
        context_.append("RETRIEVAL", retrieval.str().empty() ? "(no matches)" : retrieval.str());

        // No fresh match, but we already discussed a method → treat as follow-up.
        if (ranked.empty()) {
            if (!last_methods_.empty()) {
                debug("no new matches → continuing previous topic: " + last_topic_name_);
                return answer_follow_up(query);
            }
            return "I could not find any method in the code map matching that question. "
                   "Try using words that appear in the method name, or the method name itself.";
        }

        const bool strong = ranked.front().score >= cfg_.strong_score;
        const int limit = batch_limit_for(ranked);
        if (strong) {
            debug("strong match (score " + std::to_string(static_cast<int>(ranked.front().score)) +
                  ") → send only top method, no NOT_FOUND retry");
        }

        const std::string history = context_.history_for_prompt(cfg_.history_chars);

        size_t cursor = 0;
        std::string answer;
        std::vector<const MethodInfo*> used;
        const int batches = strong ? 1 : cfg_.max_batches;

        for (int batch = 1; batch <= batches && cursor < ranked.size(); ++batch) {
            const auto picked = next_batch(ranked, cursor, limit);
            if (picked.empty()) break;

            std::ostringstream user;
            if (!history.empty()) user << history << "\n";
            user << "Question:\n" << query << "\n\n";
            user << "Method source from the codebase:\n\n";

            used.clear();
            std::ostringstream names;
            for (size_t k = 0; k < picked.size(); ++k) {
                const auto& r = ranked[picked[k]];
                user << method_block(*r.method, static_cast<int>(k + 1), r.score) << "\n";
                used.push_back(r.method);
                names << r.method->name << (k + 1 < picked.size() ? ", " : "");
            }
            debug("batch " + std::to_string(batch) + " -> sending " + std::to_string(picked.size()) +
                  " method(s): " + names.str());
            append_saved_notes(user, used);
            context_.append("METHOD_CONTEXT", "batch " + std::to_string(batch) + "\n" + user.str());

            // Cache is pre-context for this call. The model still runs.
            answer = refine_answer(query, user.str(), /*allow_not_found=*/!strong);

            if (strong || cfg_.dry_run || !looks_like_not_found(answer)) break;
            if (cursor >= ranked.size()) {
                debug("model said NOT_FOUND and no more candidates remain");
                break;
            }
            debug("model said NOT_FOUND, trying the next batch");
        }

        last_methods_ = used;
        remember_methods(used);
        for (const MethodInfo* method : used) remember_answer(method, answer);
        return answer;
    }

    // True when the user typed a method name (or a close typo), not a question
    // like "how is invoice created".
    bool names_one_method(const std::string& query, const std::vector<RankedMethod>& ranked) const {
        if (ranked.empty()) return false;
        if (!rank_detail::identifiers_in(query).empty() && ranked.front().score >= cfg_.strong_score) return true;
        if (ranked.front().reason == "exact") return true;
        if (ranked.front().reason == "fuzzy" && ranked.front().score >= cfg_.strong_score) return true;
        return false;
    }

    // Open the most relevant methods for a question, best first.
    // Each one is sent to the model with its cache file as pre-context, then saved again.
    std::string answer_topic(const std::string& query, const std::vector<std::string>& keywords) {
        const auto ranked = rank_topic_methods(map_, keywords);
        debug("topic methods: " + std::to_string(ranked.size()));
        if (ranked.empty()) {
            return "I could not find any method in the code map matching that question. "
                   "Try using words that appear in the method name, or the method name itself.";
        }

        const int limit = std::max(1, std::min(cfg_.topic_limit, static_cast<int>(ranked.size())));
        debug("opening the " + std::to_string(limit) + " most relevant, best first");

        std::ostringstream retrieval;
        for (int i = 0; i < limit && i < cfg_.show_matches; ++i) {
            std::ostringstream line;
            line << (i + 1) << ". " << static_cast<int>(ranked[i].score) << " [topic] "
                 << describe(*ranked[i].method);
            debug(line.str());
            retrieval << line.str() << "\n";
        }
        if (static_cast<int>(ranked.size()) > limit) {
            debug(std::to_string(ranked.size() - limit) +
                  " more related methods not opened this time (RAG_TOPIC_LIMIT)");
        }
        context_.append("RETRIEVAL", retrieval.str());

        std::vector<const MethodInfo*> explored;
        std::ostringstream notes;
        size_t cursor = 0;
        int opened = 0;
        int batch = 0;
        while (opened < limit && cursor < ranked.size()) {
            const int room = limit - opened;
            const auto picked = next_batch(ranked, cursor, std::min(cfg_.batch_size, room));
            if (picked.empty()) break;
            ++batch;

            std::ostringstream user;
            user << "Question:\n" << query << "\n\n";
            user << "These methods were opened because they relate to the question, "
                 << "most relevant first. Explain what they do for this question.\n\n";
            std::vector<const MethodInfo*> used;
            std::ostringstream names;
            for (size_t k = 0; k < picked.size(); ++k) {
                const auto& r = ranked[picked[k]];
                user << method_block(*r.method, static_cast<int>(opened + k + 1), r.score) << "\n";
                used.push_back(r.method);
                names << r.method->name << (k + 1 < picked.size() ? ", " : "");
            }
            debug("topic batch " + std::to_string(batch) + " -> " + names.str());
            append_saved_notes(user, used);
            context_.append("METHOD_CONTEXT", "topic batch " + std::to_string(batch) + "\n" + user.str());

            // One pass per batch so several methods can be opened. Cache is only pre-context.
            const std::string note = refine_answer(query, user.str(), /*allow_not_found=*/false,
                                                    /*pass_limit=*/1);
            for (const MethodInfo* method : used) {
                remember_answer(method, note);
                explored.push_back(method);
            }
            notes << note << "\n\n";
            opened += static_cast<int>(picked.size());
        }

        remember_methods(explored);
        if (batch <= 1) return notes.str();

        std::ostringstream combined;
        combined << "Notes from the related methods, most relevant first:\n\n" << notes.str();
        debug("combining " + std::to_string(explored.size()) + " method notes into one answer");
        return refine_answer(query, combined.str(), /*allow_not_found=*/false, /*pass_limit=*/1);
    }

    // Follow-up like "elaborate better" — keep the same method + prior chat.
    std::string answer_follow_up(const std::string& query) {
        if (last_methods_.empty()) {
            return "I don't have a previous method in this chat yet. "
                   "Ask about a method first (e.g. what does run_reconciliation_for_customer do).";
        }

        debug("follow-up on previous topic: " + last_topic_name_ +
              " (" + std::to_string(last_methods_.size()) + " method(s))");

        std::ostringstream retrieval;
        retrieval << "follow-up; reusing previous topic\n";
        for (const auto* m : last_methods_) {
            if (m) retrieval << "  - " << describe(*m) << "\n";
        }
        context_.append("RETRIEVAL", retrieval.str());

        // Longer history so the previous answer is available to "elaborate".
        const size_t hist_chars = std::max(cfg_.history_chars, static_cast<size_t>(3500));
        const std::string history = context_.history_for_prompt(hist_chars);

        std::ostringstream user;
        if (!history.empty()) user << history << "\n";
        user << "Follow-up question:\n" << query << "\n\n";
        user << "We are still talking about: " << last_topic_name_ << "\n\n";
        user << "Method source (same as before):\n\n";

        int idx = 1;
        const MethodInfo* explained = nullptr;
        std::ostringstream names;
        for (const auto* m : last_methods_) {
            if (!m) continue;
            // One method is enough for follow-ups when the first was a strong hit.
            if (idx > 1) break;
            user << method_block(*m, idx, 0.0) << "\n";
            names << m->name;
            explained = m;
            ++idx;
        }
        if (explained) append_saved_notes(user, {explained});
        debug("follow-up -> sending " + names.str());
        context_.append("METHOD_CONTEXT", "follow-up\n" + user.str());

        const std::string answer = refine_answer(query, user.str(), /*allow_not_found=*/false);
        remember_answer(explained, answer);
        return answer;
    }

    const MethodInfo* pick_usage_target(const std::string& query) const {
        const auto identifiers = rank_detail::identifiers_in(query);
        for (const auto& id : identifiers) {
            auto ranked = rank_methods(map_, id, excluded_.keywords(id), 80.0);
            if (!ranked.empty()) return ranked.front().method;
        }

        const auto keywords = excluded_.keywords(query);
        std::vector<std::string> name_like;
        for (const auto& k : keywords) {
            if (k.size() >= 5) name_like.push_back(k);
        }
        if (!name_like.empty()) {
            std::string joined;
            for (size_t i = 0; i < name_like.size(); ++i) joined += (i ? "_" : "") + name_like[i];
            auto ranked = rank_methods(map_, joined, name_like, 90.0);
            if (!ranked.empty()) return ranked.front().method;
        }

        if (!last_methods_.empty()) return last_methods_.front();
        return nullptr;
    }

    std::string answer_usage(const std::string& query) {
        const MethodInfo* target = pick_usage_target(query);
        if (!target) {
            debug("no method name found in the question and nothing was discussed before");
            return answer_explain(query);
        }
        debug("usage target: " + describe(*target));

        const auto callers = map_.callers_of(*target);
        std::ostringstream retrieval;
        retrieval << "target: " << describe(*target) << "\n";
        for (const auto* c : callers) retrieval << "  called by: " << describe(*c) << "\n";
        context_.append("RETRIEVAL", retrieval.str());

        std::ostringstream listing;
        listing << target->name << " (" << target->location() << ")";
        if (callers.empty()) {
            listing << " has no callers in the code map.\n"
                    << "It may be called dynamically (Celery task, signal, URL route, getattr) "
                    << "or only from files outside the indexed repo.";
            last_methods_ = {target};
            return listing.str();
        }
        listing << " is called from " << callers.size() << " method(s):\n";
        for (const auto* c : callers) listing << "  - " << describe(*c) << "\n";
        debug("callers found: " + std::to_string(callers.size()));

        std::vector<RankedMethod> ranked;
        for (const auto* c : callers) ranked.push_back(RankedMethod{c, 0.0, 0.0, 1.0, "caller"});
        size_t cursor = 0;
        const auto picked = next_batch(ranked, cursor, cfg_.batch_size);

        std::ostringstream user;
        const std::string history = context_.history_for_prompt(cfg_.history_chars);
        if (!history.empty()) user << history << "\n";
        user << "Question:\n" << query << "\n\n";
        user << "Target method: " << target->name << " (" << target->location() << ")\n\n";
        append_saved_notes(user, {target});
        user << "Parent methods that call the target:\n\n";
        std::vector<const MethodInfo*> used = {target};
        for (size_t k = 0; k < picked.size(); ++k) {
            user << method_block(*ranked[picked[k]].method, static_cast<int>(k + 1), 0.0) << "\n";
            used.push_back(ranked[picked[k]].method);
        }
        if (picked.size() < callers.size()) {
            user << "(" << (callers.size() - picked.size()) << " more callers omitted for length.)\n";
        }

        debug("sending " + std::to_string(picked.size()) + " parent method(s)");
        context_.append("METHOD_CONTEXT", user.str());

        const std::string explanation = refine_answer(query, user.str(), /*allow_not_found=*/false);
        last_methods_ = used;
        return listing.str() + "\n" + explanation;
    }
};
