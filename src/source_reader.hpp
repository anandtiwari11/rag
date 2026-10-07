#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "map_loader.hpp"

// Keep the most useful parts of a long method for a small local model:
//   - first N lines  (signature, docstring, setup)
//   - last M lines   (return / final logic)
// Middle is dropped with a note. This is much smaller than a char-based cut
// of a 675-line function.
inline std::string read_method_source(const MethodMap& map,
                                      const MethodInfo& method,
                                      size_t max_chars = 6000,
                                      int head_lines = 80,
                                      int tail_lines = 40) {
    const std::string path = map.repo_root + "/" + method.file;
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open source file: " + path);
    }

    std::vector<std::string> lines;
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (line_no < method.start_line) continue;
        if (line_no > method.end_line) break;
        lines.push_back(std::to_string(line_no) + "| " + line);
    }

    auto join = [](const std::vector<std::string>& parts) {
        std::ostringstream out;
        for (const auto& p : parts) out << p << '\n';
        return out.str();
    };

    std::string text;
    if (static_cast<int>(lines.size()) <= head_lines + tail_lines) {
        text = join(lines);
    } else {
        std::vector<std::string> kept;
        kept.insert(kept.end(), lines.begin(), lines.begin() + head_lines);
        kept.push_back("... [" + std::to_string(static_cast<int>(lines.size()) - head_lines - tail_lines) +
                       " middle lines omitted for length; total " +
                       std::to_string(lines.size()) + " lines] ...");
        kept.insert(kept.end(), lines.end() - tail_lines, lines.end());
        text = join(kept);
    }

    if (text.size() <= max_chars) return text;

    const size_t head = max_chars * 2 / 3;
    const size_t tail = max_chars - head;
    return text.substr(0, head) +
           "\n... [extra character trim] ...\n" +
           text.substr(text.size() - tail);
}
