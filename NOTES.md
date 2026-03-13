# Course Notes: Build AI Agents with LangChain and LangGraph

**Course:** RAG, Tools, MCP and Production-Ready Agentic AI Systems (Python) — Udemy

---

## Section 2: LangChain Basics — Chains & Prompts

### Core Concept: LCEL (LangChain Expression Language)

LCEL uses the pipe operator `|` to compose components into chains. Each component is a **Runnable** — an object with `.invoke()`, `.stream()`, `.batch()` methods.

```python
chain = prompt_template | llm    # compose
response = chain.invoke({"information": data})  # execute
```

**Ref:** [Runnables](https://reference.langchain.com/python/langchain_core/runnables/)

### PromptTemplate

Wraps a string template with variable placeholders. LangChain validates that all `input_variables` are provided at invoke time.

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["information"],
    template="Given {information}, create a summary."
)
```

**Ref:** [Prompts](https://reference.langchain.com/python/langchain_core/prompts/)

### Chat Models

LangChain provides a unified interface across providers. All share the same `.invoke()` API:

| Provider | Class | Example |
|----------|-------|---------|
| OpenAI | `ChatOpenAI` | `ChatOpenAI(model="gpt-4o-mini", temperature=0)` |
| Ollama (local) | `ChatOllama` | `ChatOllama(model="gemma3:270m", temperature=0)` |
| Groq | `ChatGroq` | `ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)` |

`temperature=0` → deterministic responses, `temperature>0` → more creative.

**Ref:** [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)

### AIMessage

The response from a chat model is an `AIMessage` object. Access the text via `.content`.

**Ref:** [AIMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html)

### Groq Model Switching (Exercise)

- Groq provides fast inference for open-source models (Llama 4, Llama 3.3, etc.)
- Model names must match exactly what's on `console.groq.com`
- Same `invoke([messages])` pattern as other providers

---

## Section 3: Agents — Introduction

### What is an Agent?

An agent is an LLM that can **decide which tools to call** and **when to stop**. Unlike a chain (fixed sequence), an agent dynamically chooses its next action.

```python
from langchain.agents import create_agent

agent = create_agent(model=llm, tools=[TavilySearch()])
result = agent.invoke({"messages": HumanMessage(content="query")})
```

### Tavily — Web Search Tool

Tavily is a search API optimized for AI agents. Two ways to use it:

1. **Standalone SDK:** `from tavily import TavilyClient` → `tavily.search(query)`
2. **LangChain integration:** `from langchain_tavily import TavilySearch` → pass as tool to agent

The LangChain integration is preferred because the agent can call it automatically.

- **Ref:** [Tavily Quick Start](https://docs.tavily.com/sdk/python/quick-start)
- **Ref:** [LangChain Tavily Integration](https://docs.langchain.com/oss/python/integrations/providers/tavily)

### Structured Output with Pydantic

Force the agent to return data in a specific schema using `response_format`:

```python
from pydantic import BaseModel, Field

class AgentResponse(BaseModel):
    answer: str = Field(description="The agent's answer")
    sources: List[Source] = Field(default_factory=list, description="Sources used")

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
```

The agent's output is validated against the Pydantic model — guarantees structured, parseable responses.

**Ref:** [Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output)

---

## Section 4: Agent Loops — Under the Hood

This section implements the **same agent** three ways, progressively removing abstractions to understand what LangChain does internally.

### Implementation 1: LangChain Tool Calling (`.bind_tools`)

**Abstraction level:** High — LangChain handles tool schemas, message types, and tool dispatch.

```python
tools = [get_product_price, apply_discount]  # @tool decorated functions
llm_with_tools = llm.bind_tools(tools)       # attach tool schemas to LLM

# Agent loop
ai_message = llm_with_tools.invoke(messages)
tool_calls = ai_message.tool_calls  # list of {name, args, id}
```

Key pieces:
- **`@tool` decorator** auto-generates JSON schema from type hints + docstring
- **`bind_tools()`** sends tool schemas alongside every LLM call
- **Message types:** `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`
- **`ToolMessage`** feeds tool results back with matching `tool_call_id`
- **`@traceable`** decorator from LangSmith traces calls for observability

**Ref:** [Tool Calling (Ollama)](https://docs.ollama.com/capabilities/tool-calling#python), [Tool Use (Claude)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)

### Implementation 2: Raw Function Calling (Ollama SDK directly)

**Abstraction level:** Medium — uses Ollama's native tool calling, no LangChain message classes.

**Key differences from Implementation 1:**

| Aspect | LangChain (Impl 1) | Raw Ollama (Impl 2) |
|--------|-------------------|---------------------|
| Tool schemas | Auto-generated by `@tool` | **Manually written JSON** |
| LLM call | `llm_with_tools.invoke(messages)` | `ollama.chat(model, tools, messages)` |
| Messages | LangChain classes (`HumanMessage`, etc.) | **Plain dicts** (`{"role": "user", ...}`) |
| Tool call access | `tool_call.get("name")` (dict) | `tool_call.function.name` (attribute) |
| Tool execution | `tool.invoke(args)` | `tool(**args)` (direct call) |
| Tool response | `ToolMessage(content, tool_call_id)` | `{"role": "tool", "content": ...}` |
| Tracing | Automatic with LangChain | **Manual `@traceable` wrappers** |

The manual JSON schema for a tool looks like:
```python
{
    "type": "function",
    "function": {
        "name": "get_product_price",
        "description": "Look up the price of a product",
        "parameters": {
            "type": "object",
            "properties": {"product": {"type": "string", "description": "..."}},
            "required": ["product"]
        }
    }
}
```

This is exactly what `@tool` generates automatically.

### Implementation 3: Raw ReAct Prompt (Text Parsing)

**Abstraction level:** None — the LLM has **no tool-calling capability**. All agency comes from prompt engineering + regex parsing.

**ReAct (Reasoning + Acting) pattern:**
```
Thought: I need to look up the laptop price
Action: get_product_price
Action Input: laptop
Observation: 1299.99       ← injected by us, not the LLM
Thought: Now I need to apply the gold discount
Action: apply_discount
Action Input: 1299.99, gold
Observation: 1000.99       ← injected by us
Thought: I now know the final answer
Final Answer: The laptop costs $1000.99 after gold discount.
```

**Key differences from Implementation 2:**

| Aspect | Raw Ollama (Impl 2) | ReAct Prompt (Impl 3) |
|--------|---------------------|----------------------|
| Tool awareness | LLM sees JSON schemas via `tools=` | LLM sees **text descriptions in prompt** |
| Tool call format | Structured JSON response | **Free text** parsed with regex |
| History | Append message objects | **Growing string** (scratchpad) |
| Stop token | Not needed | `stop=["\nObservation"]` prevents LLM from hallucinating results |
| Parsing | Access `.tool_calls` attribute | `re.search(r"Action:\s*(.+)", output)` |
| Tool descriptions | Manual JSON or `@tool` | **`inspect.signature()` + docstring** |
| Fragility | Reliable (structured output) | **Fragile** (depends on LLM following format) |

The `stop` token is critical — it forces the LLM to pause after `Action Input:` so we can execute the real tool and inject the `Observation:` ourselves.

Tool descriptions are auto-generated from function signatures:
```python
def get_tool_descriptions(tools_dict):
    for name, func in tools_dict.items():
        original = getattr(func, "__wrapped__", func)  # bypass @traceable wrapper
        sig = inspect.signature(original)
        doc = inspect.getdoc(func)
        # produces: "get_product_price(product: str) -> float - Look up the price..."
```

### The Agent Loop Pattern (Common to All 3)

```
1. Send messages/prompt to LLM
2. Check response:
   - If final answer → return it
   - If tool call → execute tool, append result, go to 1
3. Safety: MAX_ITERATIONS limit to prevent infinite loops
```

### LangSmith Tracing

All implementations use LangSmith for observability. Enabled via `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT="ReAct Under The Hood"
```

- LangChain auto-traces chains and agents
- For raw code, use `@traceable(name="...", run_type="llm|tool")` decorator

**Ref:** [LangSmith](https://www.langchain.com/langsmith)

---

## Section 6: RAG (Retrieval-Augmented Generation)

### Why RAG?

LLMs have a knowledge cutoff and don't know your private data. RAG solves this by **retrieving relevant documents** and injecting them as context before the LLM generates an answer.

### Ingestion Pipeline

```
TextLoader → CharacterTextSplitter → OllamaEmbeddings → PineconeVectorStore
```

1. **Load:** `TextLoader` reads a text file into a `Document` object
2. **Split:** `CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)` breaks it into chunks
3. **Embed & Store:** `PineconeVectorStore.from_documents()` embeds chunks with `OllamaEmbeddings` and upserts to Pinecone

```python
loader = TextLoader("data/when_to_use_langchain.txt")
document = loader.load()
texts = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(document)
PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
```

### Retrieval Chain (without LCEL)

```python
# 1. Create retriever from vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # top-3 results

# 2. Retrieve documents
documents = retriever.invoke(query)

# 3. Format into context string
context = "\n\n".join(doc.page_content for doc in documents)

# 4. Fill prompt template
messages = prompt_template.format_messages(context=context, question=query)

# 5. Get LLM response
response = llm.invoke(messages)
```

### ChatPromptTemplate vs PromptTemplate

- `PromptTemplate` — produces a single string (used in Section 2)
- `ChatPromptTemplate` — produces a list of messages, better for chat models (used in Section 6)

```python
prompt = ChatPromptTemplate.from_template("""
    Answer the question only on the following context
    {context}
    Question: {question}
""")
```

### Raw LLM vs RAG Comparison

The code explicitly compares:
- **Option 0:** `llm.invoke([HumanMessage(content=query)])` — no context, relies on training data
- **Option 1:** Manual retrieval chain — grounded in your documents

This demonstrates the value of RAG: answers are grounded in specific, up-to-date source material.

### Next Step: LCEL Retrieval Chain

The manual approach works but has limitations noted in the code:
- No streaming support
- No async support
- Harder to compose
- More verbose

The LCEL version would look like:
```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
result = chain.invoke("When to use langchain?")
```

---

## Key Concepts Summary

| Concept | What It Does | Where Used |
|---------|-------------|------------|
| **LCEL** | Pipe operator for composing runnables | Section 2, 6 |
| **PromptTemplate** | String templates with variables | Section 2 |
| **ChatPromptTemplate** | Message-based templates | Section 6 |
| **`@tool`** | Auto-generates tool schema from function | Section 4.1 |
| **`bind_tools()`** | Attaches tool schemas to LLM | Section 4.1 |
| **ReAct** | Thought/Action/Observation loop via prompt | Section 4.3 |
| **`create_agent()`** | High-level agent factory | Section 3 |
| **`response_format`** | Pydantic-based structured output | Section 3 |
| **`@traceable`** | LangSmith observability decorator | Section 4 |
| **RAG** | Retrieve docs → inject as context → generate | Section 6 |
| **Vector Store** | Similarity search over embeddings | Section 6 |

---

## Reference Links

- [AIMessage](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html)
- [Chat Models](https://python.langchain.com/docs/concepts/chat_models/)
- [Prompts](https://reference.langchain.com/python/langchain_core/prompts/)
- [Runnables](https://reference.langchain.com/python/langchain_core/runnables/)
- [Models](https://reference.langchain.com/python/langchain/models/)
- [LangSmith](https://www.langchain.com/langsmith)
- [Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [Tavily App](https://app.tavily.com/home)
- [Tavily SDK Quick Start](https://docs.tavily.com/sdk/python/quick-start)
- [LangChain Tavily Integration](https://docs.langchain.com/oss/python/integrations/providers/tavily)
- [Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output)
- [Ollama Tool Calling](https://docs.ollama.com/capabilities/tool-calling#python)
- [Claude Tool Use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
