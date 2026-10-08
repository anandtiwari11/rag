# Codebase explorer

Ask questions about a Python repository. The app finds the matching methods, reads those lines, and asks a local Qwen model to answer from that source.

The model does not hold the whole repository. Search picks a few methods. Qwen only sees those methods plus your question.

## What happens when you ask

```text
You: what does run_reconciliation_for_customer do
        |
        v
Drop filler words (what, does, the, ...)
        |
        v
Score every method name in the map
  exact match, typo distance, or keyword overlap
        |
        v
Read that method from disk (long methods: start and end only)
        |
        v
Qwen answers from that source
        |
        v
Later passes revise the same answer so it follows your question
        |
        v
The turn is appended to utils/context.txt
```

A follow-up such as "can you elaborate" keeps the last method. It does not search again.

"Where is it used?" and `/usage <method>` read the caller list and send those parent methods to Qwen.

## Pieces

| Piece | Role |
|---|---|
| `mapping_algorithms/build_map.py` | Walk a repo, parse Python with `ast`, write `utils/map.json` |
| `utils/map.json` | Method name, file, line range, class, callers. Not committed. |
| `search_algorithms/fuzzy_name_search.hpp` | Edit distance (Levenshtein) for mistyped names |
| `src/rank_methods.hpp` | Rank names: exact, fuzzy, and keyword tokens |
| `src/rag_pipeline.hpp` | Retrieval, follow-ups, and the answer loop |
| `src/llm_client.hpp` | OpenAI-compatible call to a local server (llama.cpp, Ollama, ...) |
| `utils/context.txt` | Conversation log. Summarised after a token limit. |
| `utils/cache/` | One saved answer per method, reused the next time that method is needed. |
| `utils/excluded.txt` | Words ignored when splitting a question |

## Set up on your own machine

Do these steps in order. Use your own repository path and your own model. Nothing here has to be Qwen, and nothing has to live at a fixed folder.

### 1. Install the tools

You need:

- Git
- Python 3.10 or newer (`python3 --version`)
- A C++17 compiler (`g++` or `clang++`) and `make`
- `curl`

macOS (Homebrew):

```bash
xcode-select --install
brew install python@3.12
```

Debian or Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential python3 curl
```

The Python mapper does not need extra packages:

```bash
pip install -r requirements.txt
```

That command succeeds and installs nothing. See `requirements.txt` for the full list.

### 2. Get this project

```bash
git clone <this-repo-url>
cd <repo-folder>
```

### 3. Build the app

From the project folder:

```bash
make
```

This writes `bin/rag_ask`. On Windows, build `cmd/main.cpp` with a C++17 compiler and the include paths `-Isearch_algorithms -Isrc -Ithird_party`.

### 4. Index the repository you want to ask about

Point the mapper at your code, not this project, unless you want to index this project.

```bash
python3 -m mapping_algorithms.build_map /path/to/your/python/repo --out utils/map.json
```

Example:

```bash
python3 -m mapping_algorithms.build_map ~/work/my-service --out utils/map.json
```

You should see a line like `Indexed N methods from ...`. `utils/map.json` is created on your machine and is not part of the git clone.

Run the same command again after that repository changes. Existing method ids stay the same. New methods get the next id.

The mapper records direct Python calls. Celery `.delay()`, signals, and URL routes are often missing from "who calls this".

Check search without a model:

```bash
./bin/rag_ask --dry-run "what does your_function_name do"
```

You should see ranked method names, files, and line ranges. If you see `Cannot open map.json`, step 4 did not write `utils/map.json`.

### 5. Start your own model

This app does not load a model file. It sends HTTP requests to a local server that speaks the OpenAI chat API (`POST /v1/chat/completions`). Use any server that does that. Two common choices:

**Ollama**

```bash
# install from https://ollama.com
ollama pull qwen2.5:3b
ollama serve
```

Leave `ollama serve` running. In another terminal:

```bash
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=qwen2.5:3b
```

`LLM_MODEL` must be the name Ollama shows in `ollama list`.

**llama.cpp and a GGUF file**

A `.gguf` file is the model. `llama-server` is the program that loads it. Build or download llama.cpp, then leave this running:

```bash
cd /path/to/llama.cpp/build/bin

# macOS only, if the server cannot find its libraries:
export DYLD_LIBRARY_PATH="$(pwd):$DYLD_LIBRARY_PATH"

./llama-server \
  -m /path/to/your-model.gguf \
  -c 8192 \
  --port 8080 \
  --host 127.0.0.1
```

`-c 8192` is the context size. A long method needs a few thousand tokens. Stop the server with Ctrl+C.

In another terminal:

```bash
export LLM_BASE_URL=http://localhost:8080/v1
export LLM_MODEL=/path/to/your-model.gguf
```

Check the exact id the server expects:

```bash
curl -s "$LLM_BASE_URL/models"
```

If a later request fails with "model not found", set `LLM_MODEL` to the `id` from that JSON.

### 6. Ask a question

From the project folder, with the exports from step 5 still set:

```bash
export RAG_CONTEXT_MAX_TOKENS=1000000
export RAG_REFINE_PASSES=3
export LLM_MAX_TOKENS=1000

./bin/rag_ask
```

```text
You: what does your_function_name do
You: can you elaborate
You: /usage your_function_name
You: /quit
```

Or one question and exit:

```bash
./bin/rag_ask "what does your_function_name do"
```

The first real answer can take a minute on CPU. The model is only used after search has picked methods. `--dry-run` never calls it.

If you see `Failed to connect`, the server from step 5 is not running, or `LLM_BASE_URL` points at the wrong port.

### Commands inside the app

| Command | Effect |
|---|---|
| `/usage <method>` | Who calls that method |
| `/debug` | Toggle match scores |
| `/context` | Size of `context.txt` against the reset limit |
| `/clear` | Wipe the log and forget the previous method |
| `/quit` | Exit |

### Environment

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Local OpenAI-compatible server |
| `LLM_MODEL` | empty | Model name or GGUF path the server expects |
| `LLM_MAX_TOKENS` | `1000` | Max tokens in one model reply |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature |
| `LLM_TIMEOUT_SECONDS` | `180` | HTTP timeout |
| `RAG_CONTEXT_MAX_TOKENS` | `1000000` | Summarise `context.txt` after this many tokens |
| `RAG_REFINE_PASSES` | `3` | Model calls per answer (1 to 5) |
| `RAG_ROOT` | this project directory | Where `utils/` lives |

`RAG_REFINE_PASSES` does not force a writing style. The first call answers your question. Later calls revise that answer so it follows the question more closely. Ask for business logic, a short summary, or callers, and the reply stays on that.

## Why the model does not see the whole repo

A small local model has a short context window. A repository is far larger than that window. The map is the index. The question selects a few methods. Only those lines go into the prompt. If the right method is not selected, the model cannot use it.
 