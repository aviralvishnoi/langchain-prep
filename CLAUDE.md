# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational LangChain course project for AI engineer preparation. Organized by course sections covering chains, agents, agent loops, and RAG (Retrieval Augmented Generation).

## Commands

```bash
# Package management (uses UV, not pip)
uv sync                  # Install/sync dependencies from uv.lock
uv add <package>         # Add a dependency
uv run python <file>     # Run a script through uv

# Run any section script directly (venv must be activated)
python src/section_2/section_2.py
python src/section_4/3_raw_react_prompt.py
python src/section_6/retrieval.py

# Linting/formatting
ruff check .             # Lint
ruff format .            # Format
isort .                  # Sort imports
```

No test suite exists in this project.

## Architecture

Each `src/section_N/` directory is a standalone module corresponding to a course section:

- **section_2** — Basic LangChain chains: prompt templates + LLM composition using LCEL pipe operator (`prompt | llm`)
- **section_3** — Agents with tool integration (TavilySearch) and Pydantic structured responses
- **section_4** — Three agent loop implementations: LangChain tool calling, raw function calling, and raw ReAct prompt with manual message/iteration management
- **section_6** — RAG pipeline: `ingestion.py` (TextLoader → OllamaEmbeddings → Pinecone) and `retrieval.py` (vector retrieval chain)

Supporting data lives in `data/` (e.g., `when_to_use_langchain.txt` for RAG).

## Code Patterns

- Every script starts with `load_dotenv()` before other imports that need env vars
- Entry point pattern: `def main()` with `if __name__ == "__main__": main()`
- Logging via `loguru` (`logger.info()`, not stdlib `logging`)
- Tools use `@tool` decorator from langchain or `@traceable(run_type="tool")` for LangSmith
- LangSmith tracing is enabled by default via `.env` (`LANGSMITH_TRACING=true`)
- Multiple LLM backends: `ChatOpenAI(model="gpt-4o-mini")`, `ChatOllama`, `ChatGroq`

## Environment

- **Python 3.13** (see `.python-version`)
- **UV** package manager (not pip/poetry) — `uv.lock` is committed
- Required API keys in `.env`: `OPENAI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`, `LANGSMITH_API_KEY`
- Pinecone index name configured as `INDEX_NAME` in `.env`

## Git Conventions

- Conventional commits: `feat:`, `fix:`, `refactor:`, `chore:`
- Branch naming: `feature/<section-name>`
