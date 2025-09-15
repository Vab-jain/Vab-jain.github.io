---
title: "Book Review: Principles of Building AI Agent"
date: 2025-09-15
tags: [books, review, AI, agents]
excerpt: "A practical, short book on the general principles of building AI agents - my notes from a book that is a collection of notes"
---

I recently came across this very practical, short book on the general principles of building AI agents by Sam Bhagwat [link]. It felt more like a collection of the author's notes and practical tips compiled in a nicely structured way, rather than a heavy technical book. I really enjoyed the read; it offers a condensed introduction to the methods and standard practices of building AI agents.

This blog is basically a list of key takeaways from the book (you can see it as *"my notes from a book that is a collection of notes"* 😄).

---

## Prompting LLMs

The first couple of chapters talk about how to integrate LLMs into your pipeline. They don't go deep into the architecture, but instead treat LLMs as tools and explain how to use them effectively (via prompt engineering). It also mentions newer successor techniques to *Chain-of-thought* reasoning, such as *Chain-of-draft* and *Chain-of-preference optimization*.

> **Key tip:** Use an LLM to generate a *"Seed Prompt"* (the seed crystal approach), which you then optimize iteratively.

Good prompts are detailed and structured. Take a look at production level prompts – they are huge!

---

## Building an Agent

Agents are essentially automated programs that handle tasks, often with persistent memory and access to tools. The book broadly classifies agents as:

1. **Low Level**: binary decisions.
2. **Medium Level**: memory + tools access
3. **High Level**: can plan and complete end-to-end tasks

> **Model routing**: Use a standard SDK (like [Ollama](https://ollama.ai/)) to access models instead of juggling multiple model-specific SDKs.

> **Structured output:** LLMs are especially powerful at extracting structured data from unstructured text. They can reliably generate JSON or other structured formats.

### **Tools**

Tools are functions your agent can call for specific purposes. The list of available tools is usually included in the system prompt. Clear and detailed descriptions of tools help significantly, as do semantically meaningful names for tools and variables. Defining structured input/output is also recommended.

**Good practice:** Before coding, think through the set of tools you'll need and write down their descriptions.

### Memory

Memory defines what an agent remembers from a conversation or prior facts. It can be:

- **Working memory**: information from the current conversation
- **Hierarchical memory**: working memory + relevant long-term memory

Memory is integrated via the "context" section of the prompt. Usually, this consists of the last *x* messages, along with tool calls to fetch relevant past information (or database lookups).

### Agent Middleware

Middleware programs operate between the user and the LLM. One of the most important classes is **Guardrails**, which sanitize inputs and outputs.

- *Input guardrails*: defend against malicious prompts (jailbreaking, poisoned input) or restrict access to sensitive tools
- *Output guardrails*: filter unethical or restricted outputs

---

## Tools & MCP

Agents are only as powerful as the tools they can access. Common ones include:

- **Web-scraping**: Fragile to anti-bot programs and layout changes; requires debugging
- **Document loaders**: The [LangChain community](https://python.langchain.com/docs/integrations/document_loaders/) maintains a huge list of loaders for PDFs, social platforms, messaging services, etc.
- **Database integration**: to connect to structured knowledge sources

### MCP

The **Model Context Protocol (MCP)**, introduced by Anthropic in November 2024, is now the industry standard for integrating agents and tools. Supported by Anthropic, OpenAI, Google Gemini, and others, MCP follows a **server–client** model.

👉 Open-source MCP servers/clients include [mcp-cli](https://github.com/modelcontextprotocol/cli) and [mcp-proxy](https://github.com/modelcontextprotocol/proxy).

---

## Graph-based Workflows

Too many tools and calls can make agents unpredictable. **Workflows** define a graph-based structure where LLMs make smaller decisions at each branch rather than planning everything upfront.

Common workflow components:

- **Branching**: run parallel executions on the same input
- **Chaining**: serial execution where one tool's output feeds the next
- **Merging**: combine results from multiple branches
- **Conditionals**: if–else logic, conditional execution

> **Tip:** Keep each step semantically meaningful and simple (ideally one LLM call per step) for easier tracing.

### Streaming Updates

Since agents can take time to plan and act, **streaming updates** make the UX more engaging. A familiar example is chatbots that stream intermediate reasoning, to-do lists, or step completions.

### Observability and Tracing

Tracing lets you inspect the input/output of each tool call or LLM step. The emerging standard is [**OpenTelemetry (Otel)**](https://opentelemetry.io/). Many AI API vendors now offer UIs for workflow tracing.

---

## RAG

**Retrieval-Augmented Generation (RAG)** injects user or external data into the workflow. It's especially useful when inputs contain domain-specific or private data.

The typical RAG pipeline includes:

1. **Chunking** data
2. **Embedding** the chunks
3. **Indexing** into a vector DB
4. **Querying** by similarity (e.g., cosine similarity)
5. *(Optional)* **Reranking** (rare and time-consuming)
6. **Synthesis** via the LLM

### When to Use RAG?

You can easily over-engineer the solution with RAG. Start simple:

- Try passing full context into the LLM.
- If that doesn't work, try providing Tools to the Agent for retrieval.
- If that doesn't work, this is a good starting point to consider RAG.

Alternatives include:

- **Agentic RAG**: agent with a RAG tool
- **Reasoning Augmented Generation (ReAG)**
- **Full-context loading**

---

### Multi-Agent Systems

Multi-agent systems combine specialized agents (like managers, engineers, designers in a company). Common architectures:

- **Network**
- **Supervisor**
- **Hierarchical**
- **Custom**

---

## Evals

Evals = *tests* for LLM outputs. Since outputs are non-deterministic, evaluation is not binary but graded across a spectrum (e.g., 0–1). Each test has entropy, but overall results should converge.

| **Accuracy and Reliability** | **Understanding Context** | **Output** |
| --- | --- | --- |
| Hallucinations | Context Position | Tone Consistency |
| Faithfulness | Context Precision | Prompt Alignment |
| Content Similarity | Context Recall | Summarisation Quality |
| Completeness | Context Relevancy | Keyword Coverage |
| Answer Relevance |  |  |

*Table: List of common LLM Evaluation Metrics for Accuracy, Context Understanding, and Output Quality*

## Conclusion

Overall, this was a nice *Sunday* read. As I haven't worked directly on agents yet (which sometimes feels like #fomo 😅), I learned many new things. I especially liked the informal, to-the-point language of the book — it felt more like a discussion with the author than a lecture.

I'd definitely recommend *Principles of Building AI Agents* to anyone starting out. It's a concise, one-stop introduction to the landscape of agent design and practices.
