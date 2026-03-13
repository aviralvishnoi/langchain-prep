# Core Theory: LangChain, Agents & RAG

---

## 0. General Definitions

### LLM (Large Language Model)
A neural network trained on massive text data that predicts the next token (word/subword) in a sequence. It has no memory between calls, no access to the internet, and cannot execute code — it only generates text. Examples: GPT-4o, Llama 4, Qwen3, Claude.

### Token
The atomic unit an LLM reads and generates. Roughly ~4 characters or ~¾ of a word in English. Models have a **context window** — the maximum number of tokens they can process in a single call (input + output combined). Example: 128K tokens for GPT-4o.

### Prompt
The text input sent to an LLM. A well-crafted prompt steers the model's output. **Prompt engineering** is the practice of designing prompts to get desired behavior without changing model weights.

### Chat Model vs Completion Model
- **Completion model:** takes a raw string, outputs a string continuation (older style, e.g., GPT-3)
- **Chat model:** takes a list of messages (system/user/assistant/tool), outputs a message. All modern models (GPT-4o, Claude, Llama) are chat models. LangChain's `ChatOpenAI`, `ChatOllama`, etc. are chat model wrappers.

### Chain
A fixed sequence of steps executed in order. Input flows through each step linearly. Example: prompt → LLM → output parser. Chains are **deterministic in structure** — the steps don't change at runtime.

### Agent
An LLM that can **decide** what to do next. Unlike a chain, an agent dynamically chooses which tools to call (or none) based on the current state. The structure of execution is **determined at runtime by the LLM**. An agent is essentially an LLM inside a loop with access to tools.

### Tool
A function that an agent can invoke. The LLM doesn't execute the tool — it outputs a structured request (tool name + arguments), and the orchestrator code executes it. Tools bridge the gap between "text generation" and "real-world actions" (search, calculations, API calls, database queries).

### Tool Calling (Function Calling)
A capability built into modern LLMs where the model can output structured JSON requesting a tool invocation, instead of (or alongside) natural language text. The model is trained to emit `tool_calls` when it determines a tool is needed. This is more reliable than parsing free text (see ReAct).

### ReAct (Reasoning + Acting)
A prompting pattern where the LLM alternates between **Thought** (reasoning about what to do), **Action** (which tool to call), and **Observation** (the tool's result, injected by code). Predates native tool calling — works with any LLM by relying on the prompt format and text parsing.

### RAG (Retrieval-Augmented Generation)
A pattern that augments an LLM's knowledge by **retrieving** relevant documents from an external source (vector database, search engine) and injecting them into the prompt as context before generation. Solves the problem of LLMs not knowing private/recent data.

### Embedding
A fixed-size numerical vector (e.g., 384 or 768 floats) that represents the **semantic meaning** of a piece of text. Similar texts produce similar vectors. Used to power semantic search — you embed the query, then find the closest stored embeddings.

### Vector Store (Vector Database)
A database optimized for storing and searching embeddings by similarity (typically cosine similarity or dot product). Examples: Pinecone, Milvus, FAISS, Chroma. You store document chunks as vectors; at query time, you embed the query and retrieve the top-K most similar chunks.

### Chunking
The process of splitting large documents into smaller pieces before embedding. Necessary because embedding models have token limits, and smaller chunks yield more precise similarity matches. Key parameters: `chunk_size` (max characters per chunk) and `chunk_overlap` (shared characters between adjacent chunks for context preservation).

### LCEL (LangChain Expression Language)
LangChain's syntax for composing components using the `|` pipe operator. Each component is a Runnable, and the pipe chains them: `prompt | llm | parser`. Supports `.invoke()`, `.stream()`, `.batch()`, and async variants.

### Runnable
The base interface in LangChain. Any component that accepts input and produces output is a Runnable (prompts, LLMs, retrievers, parsers, tools). All Runnables expose the same methods (`.invoke()`, `.stream()`, etc.), making them composable.

### LangSmith
An observability platform by LangChain for tracing, debugging, and evaluating LLM applications. Records every step of a chain/agent execution with inputs, outputs, latency, and token usage.

### LangGraph
A LangChain extension for building **stateful, multi-actor** applications as graphs. Nodes are functions, edges define flow (including conditional routing and cycles). Used for complex agent architectures beyond simple loops.

### Pydantic
A Python library for data validation using type annotations. In LangChain, Pydantic models define **structured output schemas** — the LLM is constrained to return data matching the schema. Also used to define tool input schemas.

### Temperature
A parameter (0.0–2.0) controlling randomness in LLM output. `temperature=0` → deterministic, always picks the most probable token. Higher values → more creative/random responses. Use 0 for factual tasks, higher for creative tasks.

---

## 1. LangChain Architecture

LangChain is a framework of composable building blocks. Every block is a **Runnable** with a standard interface.

```
┌─────────────────────────────────────────────────────────┐
│                    LangChain Ecosystem                  │
├──────────────┬──────────────┬──────────────┬────────────┤
│ langchain-   │ langchain-   │ langchain-   │ langchain- │
│ core         │ community    │ openai       │ ollama     │
│              │              │              │            │
│ • Prompts    │ • Loaders    │ • ChatOpenAI │ • ChatOlla │
│ • Messages   │ • Splitters  │              │ • OllamaEm │
│ • Runnables  │ • Tools      │              │   beddings │
│ • Output     │              │              │            │
│   Parsers    │              │              │            │
└──────────────┴──────────────┴──────────────┴────────────┘
```

### Runnable Interface

Every LangChain component implements the Runnable protocol:

```
                    ┌──────────────┐
                    │   Runnable   │
                    ├──────────────┤
                    │ .invoke()    │  ← single input
                    │ .batch()     │  ← list of inputs
                    │ .stream()    │  ← token-by-token
                    │ .ainvoke()   │  ← async single
                    │ .astream()   │  ← async stream
                    └──────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   PromptTemplate      ChatModel      OutputParser
   Retriever           Tool           Chain
```

### LCEL (LangChain Expression Language)

LCEL lets you compose Runnables with the `|` pipe operator. Each component's output becomes the next component's input.

```
  PromptTemplate          ChatModel          StrOutputParser
  ┌───────────┐        ┌───────────┐        ┌───────────┐
  │ template + │  ───►  │   LLM     │  ───►  │  extract  │
  │ variables  │   |    │  generate │   |    │  .content │
  └───────────┘        └───────────┘        └───────────┘
       input:               input:              input:
    {"info": "..."}     ChatMessage          AIMessage
       output:             output:             output:
    ChatMessage          AIMessage             str

  chain = prompt | llm | StrOutputParser()
  result = chain.invoke({"info": "..."})
```

---

## 2. Messages — The Language of Chat Models

Chat models communicate through typed messages. Each message has a `role` and `content`.

```
┌─────────────────────────────────────────────────────┐
│                  Message Flow                        │
│                                                     │
│  SystemMessage     "You are a helpful assistant"    │
│       ▼                                             │
│  HumanMessage      "What's the price of a laptop?" │
│       ▼                                             │
│  AIMessage         "Let me look that up..."         │
│    └─ .tool_calls  [{name, args, id}]               │
│       ▼                                             │
│  ToolMessage       "1299.99"  (tool_call_id=...)    │
│       ▼                                             │
│  AIMessage         "The laptop costs $1299.99"      │
│    └─ .content     (final answer, no tool_calls)    │
└─────────────────────────────────────────────────────┘
```

| Message Type | Role | Purpose |
|-------------|------|---------|
| `SystemMessage` | system | Sets agent behavior, rules, persona |
| `HumanMessage` | user | User's query |
| `AIMessage` | assistant | LLM response — may contain `.content` and/or `.tool_calls` |
| `ToolMessage` | tool | Result from executing a tool, linked back via `tool_call_id` |

---

## 3. Tool Calling — How LLMs Use Tools

An LLM cannot execute code. It can only **request** that a tool be called by emitting structured JSON. The orchestrator (your code) executes the tool and feeds the result back.

### Tool Schema

Every tool must be described to the LLM as a JSON schema:

```
┌──────────────────────────────────────────────┐
│  What @tool decorator generates automatically │
├──────────────────────────────────────────────┤
│  {                                           │
│    "type": "function",                       │
│    "function": {                             │
│      "name": "get_product_price",            │
│      "description": "Look up the price...",  │
│      "parameters": {                         │
│        "type": "object",                     │
│        "properties": {                       │
│          "product": {                        │
│            "type": "string",                 │
│            "description": "Product name"     │
│          }                                   │
│        },                                    │
│        "required": ["product"]               │
│      }                                       │
│    }                                         │
│  }                                           │
└──────────────────────────────────────────────┘

Sources of the schema:
  • function name      → "name"
  • docstring          → "description"
  • type hints (str)   → "type": "string"
  • parameter names    → "properties" keys
```

### Three Levels of Tool Calling

```
Level 3 (Highest Abstraction)         Level 2 (Medium)              Level 1 (No Abstraction)
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│  LangChain @tool     │     │  Raw Ollama SDK      │     │  ReAct Prompt        │
│  + bind_tools()      │     │  + manual JSON       │     │  + regex parsing     │
├──────────────────────┤     ├──────────────────────┤     ├──────────────────────┤
│                      │     │                      │     │                      │
│ • @tool auto-creates │     │ • You write JSON     │     │ • Tools described    │
│   JSON schema        │     │   schemas by hand    │     │   as plain text in   │
│                      │     │                      │     │   the prompt         │
│ • .bind_tools()      │     │ • Pass tools= to     │     │                      │
│   attaches schemas   │     │   ollama.chat()      │     │ • No tools= param   │
│                      │     │                      │     │   at all             │
│ • LangChain message  │     │ • Plain dict         │     │                      │
│   classes            │     │   messages           │     │ • Single prompt      │
│                      │     │                      │     │   string (scratchpad)│
│ • .tool_calls is a   │     │ • .tool_calls uses   │     │                      │
│   list of dicts      │     │   attribute access   │     │ • Regex extracts     │
│                      │     │   (.function.name)   │     │   Action/Action Input│
│ • tool.invoke(args)  │     │ • func(**args)       │     │   from raw text      │
│                      │     │   direct call        │     │                      │
│ • Auto-traced in     │     │ • Manual @traceable  │     │ • stop=["\nObserv.."]│
│   LangSmith          │     │   wrappers needed    │     │   prevents LLM from  │
│                      │     │                      │     │   faking results     │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

---

## 4. The Agent Loop

An agent is just a **loop** that alternates between LLM reasoning and tool execution until the LLM decides it has enough information to answer.

```
                    ┌─────────────┐
                    │  User Query │
                    └──────┬──────┘
                           ▼
                ┌──────────────────┐
                │  Prepare Messages │
                │  (system + human) │
                └────────┬─────────┘
                         ▼
              ┌─────────────────────┐
         ┌───►│   Invoke LLM        │
         │    └────────┬────────────┘
         │             ▼
         │    ┌─────────────────────┐
         │    │  Has tool_calls?    │
         │    └───┬─────────────┬───┘
         │        │ YES         │ NO
         │        ▼             ▼
         │  ┌───────────┐  ┌──────────────┐
         │  │ Execute    │  │ Return       │
         │  │ Tool       │  │ Final Answer │
         │  └─────┬─────┘  └──────────────┘
         │        │
         │        ▼
         │  ┌───────────────┐
         │  │ Append:        │
         │  │ • AIMessage    │
         │  │ • ToolMessage  │
         │  └───────┬───────┘
         │          │
         │          ▼
         │  ┌───────────────┐
         │  │ iteration <   │──── NO ───► Error: Max iterations
         │  │ MAX?          │
         │  └───────┬───────┘
         │      YES │
         └──────────┘
```

### Key Design Decisions in the Loop

**1. One tool per iteration** — Process only `tool_calls[0]` to maintain control and debuggability. The LLM may request multiple tools, but executing one at a time keeps the loop predictable.

**2. MAX_ITERATIONS guard** — Prevents infinite loops if the LLM never produces a final answer. Typical value: 10.

**3. Message accumulation** — Every iteration appends the AIMessage + ToolMessage to the conversation. The LLM sees the full history on each call, building up context.

---

## 5. ReAct Pattern (Reasoning + Acting)

ReAct is a prompting strategy where the LLM explicitly writes out its reasoning before taking action. It was introduced before native tool calling existed and still works with any LLM.

```
┌────────────────────────────────────────────────────────────┐
│                     ReAct Format                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Question: What is the laptop price after gold discount?   │
│                                                            │
│  Thought: I need to find the laptop price first.          │
│  Action: get_product_price                      ◄── LLM   │
│  Action Input: laptop                           ◄── LLM   │
│  ~~~~~~~~ STOP TOKEN ("\nObservation") ~~~~~~~~            │
│  Observation: 1299.99                           ◄── CODE  │
│                                                            │
│  Thought: Now I need to apply the gold discount.          │
│  Action: apply_discount                         ◄── LLM   │
│  Action Input: 1299.99, gold                    ◄── LLM   │
│  ~~~~~~~~ STOP TOKEN ("\nObservation") ~~~~~~~~            │
│  Observation: 1000.99                           ◄── CODE  │
│                                                            │
│  Thought: I now know the final answer.                    │
│  Final Answer: The laptop costs $1000.99        ◄── LLM   │
│                                                            │
└────────────────────────────────────────────────────────────┘

Legend:  LLM = generated by the model
        CODE = injected by your orchestrator code
```

### Stop Token Mechanism

```
Without stop token:                    With stop token:
┌─────────────────────────┐           ┌─────────────────────────┐
│ Thought: look up price  │           │ Thought: look up price  │
│ Action: get_product_... │           │ Action: get_product_... │
│ Action Input: laptop    │           │ Action Input: laptop    │
│ Observation: 999.99 ◄─ HALLUCINATED│ █ STOPPED HERE           │
│ Thought: I know the ... │           │                         │
│ Final Answer: $999.99   │ WRONG!    │ (we inject real result) │
└─────────────────────────┘           └─────────────────────────┘
```

The stop token `["\nObservation"]` is **critical** — it prevents the LLM from hallucinating tool results and forces control back to your code.

### Scratchpad vs Message List

```
Structured Tool Calling (Impl 1 & 2):     ReAct (Impl 3):
┌────────────────────────┐                ┌────────────────────────┐
│ messages = [           │                │ prompt = react_template│
│   SystemMessage(...)   │                │ scratchpad = ""        │
│   HumanMessage(...)    │                │                        │
│   AIMessage(...)       │                │ Each iteration:        │
│   ToolMessage(...)     │                │ full_prompt = prompt   │
│   AIMessage(...)       │                │             + scratchpad│
│   ToolMessage(...)     │                │                        │
│   ...                  │                │ scratchpad +=          │
│ ]                      │                │   output +             │
│                        │                │   "\nObservation: " +  │
│ Typed objects          │                │   result +             │
│ Each has a role        │                │   "\nThought:"         │
│ Tool call IDs tracked  │                │                        │
└────────────────────────┘                │ One growing string     │
                                          │ No types or IDs        │
                                          └────────────────────────┘
```

---

## 6. RAG (Retrieval-Augmented Generation)

### The Problem RAG Solves

```
Without RAG:                              With RAG:
┌──────────┐                             ┌──────────┐
│   User   │──── "When to use            │   User   │──── "When to use
│          │      langchain?" ──┐        │          │      langchain?" ──┐
└──────────┘                    ▼        └──────────┘                    ▼
                         ┌──────────┐                          ┌─────────────────┐
                         │   LLM    │                          │    Retriever     │
                         │          │                          │  (vector search) │
                         │ Only knows│                          └────────┬────────┘
                         │ training │                                   │
                         │ data     │                            relevant chunks
                         └────┬─────┘                                   ▼
                              │                                ┌─────────────────┐
                              ▼                                │   LLM + Context │
                      Generic/outdated                         │                 │
                      answer                                   │ "Based on the   │
                                                               │  provided docs..│
                                                               └────────┬────────┘
                                                                        │
                                                                        ▼
                                                               Grounded, accurate
                                                               answer
```

### Ingestion Pipeline

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Raw Document│     │  Text Splitter   │     │   Embedding      │     │  Vector Store │
│              │     │                  │     │   Model          │     │  (Pinecone)   │
│  .txt file   │────►│ CharacterText    │────►│ OllamaEmbeddings │────►│               │
│  .pdf        │     │ Splitter         │     │ (qwen3-embedding │     │  Stores       │
│  .csv        │     │                  │     │  :0.6b)          │     │  vectors +    │
│              │     │ chunk_size=1000  │     │                  │     │  metadata     │
└──────────────┘     │ chunk_overlap=0  │     │ text → [0.12,    │     │               │
                     └──────────────────┘     │  -0.45, 0.78..]  │     └───────────────┘
                                              └──────────────────┘
      1 document          N chunks              N vectors              N records
```

### Why Chunk?

```
┌─────────────────────────────────────────────────────────┐
│  Original Document (5000 tokens)                        │
│  "LangChain is an open-source framework... [very long]" │
└─────────────────────────────────────────────────────────┘
                          │
                    Text Splitter
                    chunk_size=1000
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ Chunk 1   │   │ Chunk 2   │   │ Chunk 3   │   ...
    │ "LangChain│   │ "When you │   │ "When you │
    │  is an    │   │  should   │   │  should   │
    │  open..." │   │  use it"  │   │  NOT use" │
    └───────────┘   └───────────┘   └───────────┘

Benefits:
• Embedding models have token limits
• Smaller chunks = more precise similarity matches
• Only relevant chunks are sent to LLM (saves tokens)

chunk_overlap: characters shared between adjacent chunks
  to preserve context at boundaries (0 = no overlap)
```

### Retrieval Pipeline

```
┌───────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Query    │     │  Embed Query     │     │  Vector Search  │
│           │────►│  Same model as   │────►│  (cosine sim)   │
│ "When to  │     │  ingestion       │     │                 │
│  use      │     │                  │     │  k=3 (top 3     │
│  langchain│     │  query → [0.11,  │     │   matches)      │
│  ?"       │     │   -0.43, 0.79..] │     │                 │
└───────────┘     └──────────────────┘     └────────┬────────┘
                                                    │
                                              3 Documents
                                                    │
                                                    ▼
                                           ┌────────────────┐
                                           │ Format Context  │
                                           │                 │
                                           │ doc1.page_content│
                                           │ + "\n\n" +      │
                                           │ doc2.page_content│
                                           │ + "\n\n" +      │
                                           │ doc3.page_content│
                                           └───────┬────────┘
                                                   │
                                                   ▼
                                           ┌────────────────┐
                                           │ Prompt Template │
                                           │                 │
                                           │ "Answer based   │
                                           │  on: {context}  │
                                           │  Question:      │
                                           │  {question}"    │
                                           └───────┬────────┘
                                                   │
                                                   ▼
                                           ┌────────────────┐
                                           │     LLM        │
                                           │  (qwen3:1.7b)  │
                                           └───────┬────────┘
                                                   │
                                                   ▼
                                            Grounded Answer
```

### Embeddings — The Core of Semantic Search

```
Traditional keyword search:          Embedding/semantic search:

"When to use LangChain"             "When to use LangChain"
        │                                    │
  exact string matching               maps to vector space
        │                                    │
  ✗ "LangChain is ideal for..."      ✓ "LangChain is ideal for..."
    (no exact match for "when")         (semantically similar)
  ✗ "Use LangChain when you need.."  ✓ "Use LangChain when you need..."
    (different word order)              (close in vector space)
  ✓ "When to use LangChain"          ✓ "When to use LangChain"
    (exact match only)                  (also close, obviously)
```

Embedding models convert text into high-dimensional vectors (e.g., 384 or 768 dimensions). Texts with similar **meaning** end up close together in vector space, regardless of exact wording.

---

## 7. Structured Output

Forces the LLM to return data matching a Pydantic schema instead of free-form text.

```
Without structured output:              With structured output:
┌─────────────────────────┐            ┌─────────────────────────┐
│ "The capital of France  │            │ {                       │
│  is Paris. It's known   │            │   "answer": "Paris",   │
│  for the Eiffel Tower." │            │   "sources": [         │
│                         │            │     {"url": "https://  │
│  ↓ parse this? how?     │            │       en.wikipedia..."}│
└─────────────────────────┘            │   ]                    │
                                       │ }                      │
                                       │                        │
                                       │  ↓ already structured! │
                                       └─────────────────────────┘

Pydantic model:
class AgentResponse(BaseModel):
    answer: str
    sources: List[Source]

Usage:
  agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
```

---

## 8. LangSmith — Observability & Tracing

LangSmith records every step of your chain/agent for debugging and evaluation.

```
┌──────────────────────────────────────────────────────────┐
│  LangSmith Trace View                                    │
│                                                          │
│  ► Ollama Agent Loop                        3.2s         │
│    ├── Ollama Chat (LLM)                    0.8s         │
│    │   └─ input: messages[...]                           │
│    │   └─ output: tool_call(get_product_price)           │
│    ├── get_product_price (Tool)             0.001s       │
│    │   └─ input: product="laptop"                        │
│    │   └─ output: 1299.99                                │
│    ├── Ollama Chat (LLM)                    0.9s         │
│    │   └─ output: tool_call(apply_discount)              │
│    ├── apply_discount (Tool)                0.001s       │
│    │   └─ input: price=1299.99, tier="gold"              │
│    │   └─ output: 1000.99                                │
│    └── Ollama Chat (LLM)                    0.7s         │
│        └─ output: "The laptop costs $1000.99"            │
└──────────────────────────────────────────────────────────┘

Tracing methods:
  • LangChain chains/agents: auto-traced
  • Raw functions: @traceable(name="...", run_type="tool|llm")
  • Enabled via: LANGSMITH_TRACING=true in .env
```

---

## 9. Model Provider Comparison

```
┌─────────────┬──────────────────┬───────────┬────────────────────────┐
│  Provider   │  Class           │  Where    │  Best For              │
├─────────────┼──────────────────┼───────────┼────────────────────────┤
│  OpenAI     │  ChatOpenAI      │  Cloud    │  Most capable,         │
│             │                  │  (API)    │  best tool calling     │
├─────────────┼──────────────────┼───────────┼────────────────────────┤
│  Ollama     │  ChatOllama      │  Local    │  Free, private,        │
│             │                  │  machine  │  no API key needed     │
├─────────────┼──────────────────┼───────────┼────────────────────────┤
│  Groq       │  ChatGroq        │  Cloud    │  Fastest inference,    │
│             │                  │  (API)    │  open-source models    │
└─────────────┴──────────────────┴───────────┴────────────────────────┘

All share the same interface:
  llm = ChatXxx(model="...", temperature=0)
  response = llm.invoke(messages)
  text = response.content
```
