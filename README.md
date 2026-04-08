# 🚀 CodeForge: Agentic Multi-Agent Code Generation

**Live Demo**: [CodeForge Alpha](http://codefo-publi-cnig9v7f5w21-523519748.ap-south-1.elb.amazonaws.com/)

CodeForge is a professional-grade AI code-generation platform built on **LangGraph**. It orchestrates a specialized team of agents to transform natural language prompts into production-ready web applications, complete with recursive self-correction and a sophisticated long-term memory system.

---

## 🏗️ The Multi-Agent Architecture

Unlike simple "one-shot" generators, CodeForge uses a structured pipeline where each agent has a distinct engineering role:

| Agent | Responsibility |
| :--- | :--- |
| **Planner** | Breaks the prompt into a technical roadmap. Recalls past mistakes to avoid regression. |
| **Architect** | Defines file structures, function signatures, and dependency graphs. |
| **Coder** | Implements code file by file, utilizing RAG for implementation best practices. |
| **Reviewer** | Performs static code analysis, identifies bugs, and assigns quality scores. |
| **Fixer** | Targeted debugging based on Reviewer feedback. |

---

## 🧠 Episodic Memory: Learning from Experience

The "secret sauce" of CodeForge is its **Episodic Memory System**, powered by **PostgreSQL** and **pgvector**. This system allows the agentic workflow to improve its performance the more it is used.

### How it Works
1.  **Post-Run Analysis**: After every generation, the *Reviewer* identifies issues (high, medium, or low severity).
2.  **Storage**: These issues, along with the original user intent (embedded via OpenAI's `text-embedding-3-small`), are stored in a vector database.
3.  **Semantic Recall**: When a new prompt is received, the *Planner* calls the `recall_past_mistakes()` tool. It performs a cosine-similarity search to find "episodes" of similar projects where mistakes were made.
4.  **Proactive Mitigation**: The Planner is forced to include explicit guardrails for any high-severity mistakes encountered in the past.

### Performance Impact
Instead of repeating errors (e.g., missing a specific dependency or a common logic bug in a certain tech stack), the agent **learns**. This results in:
*   **Reduced Iteration Cycles**: Code often works on the first try because past "lessons" are applied upfront.
*   **Knowledge Persistence**: The system naturally builds a "library of pitfalls" for specific frameworks.
*   **Self-Healing Capabilities**: Regressive bugs are systematically eliminated as the memory database grows.

---

## 📚 Retrieval Augmented Generation (RAG)

While Episodic Memory handles **experience**, RAG handles **expertise**. CodeForge integrates a dedicated RAG pipeline to ensure the code produced follows modern engineering standards.

### The Knowledge Base
The `Agents/RAG/Data` directory serves as a high-density knowledge center where PDF documentation, style guides, and framework best practices are ingested.

### Dynamic Tooling
The **Coder** agent utilizes the `rag_query()` tool during the implementation phase. For example, if building a FastAPI route, the Coder queries the vector store for implementation patterns specific to that task.

### Why Results Vary
Because RAG performs semantic retrieval, the "context" provided to the agent is non-deterministic:
*   **Contextual Nuance**: Slight variations in the Coder's query can pull different "best practice" chunks.
*   **Information Density**: The agent may receive different snippets of documentation depending on the similarity threshold, leading to diverse but valid implementation strategies (e.g., using different middleware for the same problem).
*   **Quality Variance**: The presence or absence of specific documentation in the RAG store directly dictates the "seniority" of the code produced.

---

## 🛠️ Tech Stack

*   **Orchestration**: LangGraph (LangChain)
*   **Backend**: FastAPI with SSE (Server-Sent Events) for real-time streaming
*   **LLMs**: OpenAI GPT-4o / Claude 3.5 Sonnet
*   **Database**: PostgreSQL + pgvector (for memory and knowledge storage)
*   **Embeddings**: OpenAI `text-embedding-3-small`
*   **Containerization**: Docker (Multi-stage builds)

---

## 🚦 Getting Started

### Prerequisites
- Python 3.10+
- PostgreSQL with `pgvector`
- OpenAI API Key

### Installation

1.  **Clone and Install Dependencies**:
    ```bash
    git clone https://github.com/your-repo/CodeForge.git
    cd CodeForge
    pip install -r requirements.txt
    ```

2.  **Environment Setup**:
    Create a `.env` file based on `.env.example`:
    ```env
    OPENAI_API_KEY=your_key
    DATABASE_URL=postgresql://user:pass@localhost:5432/codeforge
    ```

3.  **Run the Application**:
    ```bash
    python main.py
    ```

---

## 🚀 Deployment & Cloud Infrastructure

CodeForge is built for scale and utilizes AWS native services for high availability and secure data handling.

### AWS ECS (Elastic Container Service)
The core application is containerized using **Docker** and deployed on **AWS ECS**.
- **Orchestration**: ECS manages the lifecycle of our multi-agent workers, ensuring they are always available to handle generation requests.
- **Scalability**: The system is designed to scale horizontally, allowing multiple agentic workflows to run in parallel across a distributed cluster.
- **Resilience**: Integrated with AWS Service Connect for reliable internal networking between the API and the PostgreSQL/pgvector database.

### AWS S3 (Simple Storage Service)
All generated projects are packaged into secure ZIP archives and offloaded to **AWS S3**.
- **Secure Downloads**: We leverage **S3 Presigned URLs** (24-hour expiration) to ensure that your generated code is only accessible to you.
- **Persistence**: While the agent's working directory is ephemeral, S3 provides durable storage for all successful builds.
- **Efficiency**: Offloading heavy binary assets (ZIPs) to S3 keeps the application containers lightweight and responsive.

### Getting Started Locally
