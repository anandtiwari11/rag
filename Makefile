# Build the C++ codebase explorer pipeline
#
#   make
#   export LLM_BASE_URL=http://localhost:11434/v1
#   export LLM_MODEL=your-qwen-model
#   ./bin/rag_ask "run_reconciliatio_for_customer"

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
INCLUDES  = -Isearch_algorithms -Isrc -Ithird_party

SRC = cmd/main.cpp
BIN = bin/rag_ask

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRC)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(BIN) $(SRC)

run: $(BIN)
	$(BIN) $(QUERY)

clean:
	rm -f $(BIN)
