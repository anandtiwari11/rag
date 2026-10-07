#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"

struct CallRef {
    std::string name;
    int id = -1;  // -1 when the mapper could not resolve the callee
};

struct MethodInfo {
    int id = 0;
    std::string name;
    std::string file;
    std::string folder;
    int start_line = 0;
    int end_line = 0;
    std::string kind;
    std::string parent_class;
    std::string parent_function;
    std::string qualified_name;
    std::vector<CallRef> calls;    // what this method calls
    std::vector<int> called_by;    // parent methods that call this one

    int line_count() const { return end_line - start_line + 1; }

    std::string location() const {
        return file + ":" + std::to_string(start_line) + "-" + std::to_string(end_line);
    }
};

// In-memory copy of utils/map.json (the "phonebook" of the repo).
class MethodMap {
public:
    std::string repo_root;
    std::vector<MethodInfo> methods;

    static MethodMap load(const std::string& path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Cannot open map.json: " + path);
        }
        nlohmann::json j;
        in >> j;

        MethodMap map;
        map.repo_root = j.value("repo_root", "");
        for (const auto& item : j.at("methods")) {
            MethodInfo m;
            m.id = item.value("id", 0);
            m.name = item.value("name", "");
            m.file = item.value("file", "");
            m.folder = item.value("folder", "");
            m.start_line = item.value("start_line", 0);
            m.end_line = item.value("end_line", 0);
            m.kind = item.value("kind", "");
            m.parent_class = string_or_empty(item, "parent_class");
            m.parent_function = string_or_empty(item, "parent_function");
            m.qualified_name = item.value("qualified_name", m.name);

            if (item.contains("calls") && item["calls"].is_array()) {
                for (const auto& c : item["calls"]) {
                    CallRef ref;
                    ref.name = c.value("name", "");
                    ref.id = (c.contains("id") && !c["id"].is_null()) ? c["id"].get<int>() : -1;
                    m.calls.push_back(std::move(ref));
                }
            }
            if (item.contains("called_by") && item["called_by"].is_array()) {
                for (const auto& id : item["called_by"]) {
                    if (id.is_number_integer()) m.called_by.push_back(id.get<int>());
                }
            }
            map.methods.push_back(std::move(m));
        }

        for (size_t i = 0; i < map.methods.size(); ++i) {
            map.id_index_[map.methods[i].id] = i;
        }
        return map;
    }

    const MethodInfo* find_by_id(int id) const {
        auto it = id_index_.find(id);
        if (it == id_index_.end()) return nullptr;
        return &methods[it->second];
    }

    std::vector<const MethodInfo*> find_by_name(const std::string& name) const {
        std::vector<const MethodInfo*> out;
        for (const auto& m : methods) {
            if (m.name == name) out.push_back(&m);
        }
        return out;
    }

    std::vector<const MethodInfo*> callers_of(const MethodInfo& m) const {
        std::vector<const MethodInfo*> out;
        for (int id : m.called_by) {
            if (const MethodInfo* p = find_by_id(id)) out.push_back(p);
        }
        return out;
    }

private:
    std::unordered_map<int, size_t> id_index_;

    static std::string string_or_empty(const nlohmann::json& item, const char* key) {
        if (!item.contains(key) || item[key].is_null()) return "";
        return item[key].get<std::string>();
    }
};
