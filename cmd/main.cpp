#include <cstdlib>
#include <iostream>
#include <string>
#include "context_store.hpp"
#include "excluded.hpp"
#include "llm_client.hpp"
#include "map_loader.hpp"
#include "rag_pipeline.hpp"

namespace {

std::string project_root() {
    const char* env = std::getenv("RAG_ROOT");
    return (env && *env) ? std::string(env) : "/Users/dehaat/Anand/RAG";
}

int env_int(const char* name, int fallback, int lo, int hi) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    int n = std::atoi(value);
    if (n < lo) return lo;
    if (n > hi) return hi;
    return n;
}

void apply_env(RagConfig& cfg) {
    cfg.context_max_tokens = env_int("RAG_CONTEXT_MAX_TOKENS", cfg.context_max_tokens, 1000, 5000000);
    cfg.refine_passes = env_int("RAG_REFINE_PASSES", cfg.refine_passes, 1, 5);
    cfg.topic_limit = env_int("RAG_TOPIC_LIMIT", cfg.topic_limit, 1, 12);
}

void print_help() {
    std::cout << "Commands:\n"
                 "  /usage <method>   who calls this method (sends parents to the model)\n"
                 "  /debug            toggle retrieval details\n"
                 "  /context          show context.txt size\n"
                 "  /clear            wipe context.txt (recommended after failed runs)\n"
                 "  /help             this help\n"
                 "  /quit             exit\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string root = project_root();
        RagConfig cfg;
        apply_env(cfg);
        cfg.notes_dir = root + "/utils/cache";
        std::string query;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dry-run") cfg.dry_run = true;
            else if (arg == "--quiet") cfg.debug = false;
            else if (arg == "--help" || arg == "-h") { print_help(); return 0; }
            else { if (!query.empty()) query.push_back(' '); query += arg; }
        }

        MethodMap map = MethodMap::load(root + "/utils/map.json");
        ExcludedWords excluded(root + "/utils/excluded.txt");
        LlmClient llm;
        ContextStore context(root + "/utils/context.txt", cfg.context_max_tokens);
        RagPipeline rag(map, excluded, llm, context, cfg);

        std::cout << "loaded " << map.methods.size() << " methods from " << map.repo_root << "\n";
        std::cout << "context resets after ~" << cfg.context_max_tokens
                  << " tokens (RAG_CONTEXT_MAX_TOKENS)\n";
        std::cout << "llm refine passes: " << cfg.refine_passes << " (RAG_REFINE_PASSES)\n";
        std::cout << "topic questions open the top " << cfg.topic_limit
                  << " related methods (RAG_TOPIC_LIMIT)\n";
        if (!llm.configured() && !cfg.dry_run) {
            std::cout << "LLM_MODEL is not set. Set it (e.g. export LLM_MODEL=qwen2.5:3b) "
                         "or use --dry-run to inspect retrieval only.\n";
        } else if (!cfg.dry_run) {
            std::cout << "model: " << llm.model() << " @ " << llm.base_url() << "\n";
        }

        if (!query.empty()) {
            std::cout << "\nYou: " << query << "\n";
            std::cout << "\nAI: " << rag.ask(query) << "\n";
            return 0;
        }

        print_help();
        std::string line;
        for (;;) {
            std::cout << "\nYou: ";
            if (!std::getline(std::cin, line)) break;
            line = LlmClient::trim(line);
            if (line.empty()) continue;

            if (line == "/quit" || line == "/exit") break;
            if (line == "/help") { print_help(); continue; }
            if (line == "/debug") {
                rag.config().debug = !rag.config().debug;
                std::cout << "debug " << (rag.config().debug ? "on" : "off") << "\n";
                continue;
            }
            if (line == "/context") {
                std::cout << "context.txt ~" << context.estimate_tokens() << " tokens (limit "
                          << context.max_tokens() << ")\n";
                continue;
            }
            if (line == "/clear") {
                context.clear();
                rag.clear_topic();
                std::cout << "context.txt cleared (previous method topic forgotten)\n";
                continue;
            }
            if (line.rfind("/usage ", 0) == 0) {
                std::cout << "\nAI: " << rag.usage_of(LlmClient::trim(line.substr(7))) << "\n";
                continue;
            }

            try {
                std::cout << "\nAI: " << rag.ask(line) << "\n";
            } catch (const std::exception& e) {
                std::cout << "\nError: " << e.what() << "\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
