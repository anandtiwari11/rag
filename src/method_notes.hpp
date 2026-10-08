#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "map_loader.hpp"

// One text file per explored method, under utils/cache/.
// The next time that method is sent to the model, the saved answer goes with it.
class MethodNotes {
public:
    explicit MethodNotes(std::string dir, const MethodMap& map) : dir_(std::move(dir)), map_(&map) {}

    std::string path_for(const MethodInfo& method) const {
        std::string name = sanitize(method.name);
        if (name.empty()) name = "method";
        if (map_->find_by_name(method.name).size() > 1) {
            name += "__" + std::to_string(method.id);
        }
        return dir_ + "/" + name + ".txt";
    }

    // Empty if missing, unreadable, or written for a different line range.
    std::string load(const MethodInfo& method) const {
        std::ifstream in(path_for(method));
        if (!in) return "";
        std::string line;
        int id = -1;
        std::string file;
        std::string lines;
        while (std::getline(in, line)) {
            if (line == "---") break;
            if (line.rfind("id: ", 0) == 0) id = std::atoi(line.c_str() + 4);
            else if (line.rfind("file: ", 0) == 0) file = line.substr(6);
            else if (line.rfind("lines: ", 0) == 0) lines = line.substr(7);
        }
        const std::string expected = std::to_string(method.start_line) + "-" + std::to_string(method.end_line);
        if (id != method.id || file != method.file || lines != expected) return "";
        std::ostringstream body;
        body << in.rdbuf();
        std::string text = body.str();
        while (!text.empty() && (text.front() == '\n' || text.front() == '\r')) text.erase(text.begin());
        return text;
    }

    void save(const MethodInfo& method, const std::string& answer) const {
        if (dir_.empty() || answer.empty()) return;
        std::filesystem::create_directories(dir_);
        std::ofstream out(path_for(method), std::ios::trunc);
        if (!out) return;
        out << "id: " << method.id << "\n";
        out << "name: " << method.name << "\n";
        out << "file: " << method.file << "\n";
        out << "lines: " << method.start_line << "-" << method.end_line << "\n";
        out << "---\n";
        out << answer << "\n";
    }

private:
    std::string dir_;
    const MethodMap* map_;

    static std::string sanitize(const std::string& name) {
        std::string out;
        for (char c : name) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') {
                out.push_back(c);
            } else {
                out.push_back('_');
            }
        }
        return out;
    }
};
