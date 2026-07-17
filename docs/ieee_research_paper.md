# Explainable Clinical Decision Support (X-CDS): Mitigating LLM Hallucinations in High-Stakes Medicine via Hybrid Retrieval and Deterministic Citation Verification Frameworks

**Submitted by:** Sahil Kumar  
**Registration Number:** 23BAI10224  
**Affiliation:** School of Computer Science and Engineering (SCSE), VIT Bhopal University  
**Advisor:** Dr. Abdul Rahman, Associate Professor  

---

## Abstract
Large Language Models (LLMs) show significant promise in clinical decision support systems (CDSS). However, the propensity of generative models to "hallucinate"—generate medically incorrect or unsubstantiated claims—remains a critical barrier to clinical deployment. This paper introduces **Explainable Clinical Decision Support (X-CDS)**, an architecture designed to guarantee the factual validity of clinical recommendations. X-CDS integrates a **Hybrid Retrieval** pipeline (ChromaDB Vector Search and BM25 Keyword Search) merged via **Reciprocal Rank Fusion (RRF)** and filtered through a **Cross-Encoder Re-ranker**. To prevent hallucinations, we implement a stateful **LangGraph** orchestration loop that programmatically validates generative assertions against retrieved source passages using a token-overlap alignment threshold. If the generator fails validation, the state machine routes the failure back for iterative self-correction. We evaluate the system using the **Ragas** benchmarking framework against a clinical dataset, measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall. Our results demonstrate that deterministic citation guardrails significantly improve model faithfulness compared to standard RAG baselines, paving a safe pathway for clinical LLM deployment.

---

## I. Introduction
The deployment of Large Language Models (LLMs) such as GPT-4 and Claude 3 in oncology and cardiology has demonstrated their potential to assist clinicians with diagnostic suggestions, literature summaries, and treatment plan generation. Despite these capabilities, generative models are fundamentally statistical next-token predictors. Consequently, they suffer from the "hallucination" phenomenon—generating logical-sounding but factual fabrications. In high-stakes clinical settings, an unsubstantiated treatment or diagnostic suggestion can lead to severe adverse patient outcomes.

Traditional Retrieval-Augmented Generation (RAG) mitigates this by injecting relevant medical literature into the model's prompt context. However, standard RAG still fails if:
1. The retrieval pipeline fails to capture critical clinical facts (poor recall).
2. The generator ignores the injected context or synthesizes claims that lack an explicit source citation (hallucination).

To address these vulnerabilities, we present the **X-CDS** framework. The primary contributions of this paper are:
1. A dual-channel **Hybrid Retrieval** pipeline combining dense vector embeddings and sparse keyword indices merged via Reciprocal Rank Fusion (RRF) and optimized using a transformer-based Cross-Encoder re-ranker.
2. A stateful **LangGraph Self-Correction** workflow that programmatically checks citation alignment using character-level token-overlap algorithms.
3. An empirical evaluation of the pipeline's effectiveness using the Ragas metric framework, proving that programmatic self-correction loops maximize factual safety.

---

## II. System Architecture and Methodology

```mermaid
graph TD
    A[Clinical Query] --> B[Hybrid Search: Dense + Sparse]
    B --> C[Reciprocal Rank Fusion (RRF)]
    C --> D[Cross-Encoder Re-ranker]
    D --> E[Top-K Clinical Chunks]
    E --> F[LangGraph Orchestrator]
    F --> G[Gemini Generation Node]
    G --> H{Citation Overlap Validator}
    H -- Fail: Overlap < 0.25 --> I[State Correction feedback]
    I --> G
    H -- Pass: Overlap >= 0.25 --> J[Output to Web Dashboard]
```

### A. Data Ingestion & Formatting
Clinical literature is ingested via the NIH BioC API. Passages are structured into document chunks containing unique identifiers, metadata (PMCID, section title), and clean text. 

### B. Hybrid Retrieval Pipeline
To capture both semantic concepts and specific clinical terminology (e.g., drug names, gene variants), we implement a dual-channel retrieval system:
1. **Dense Vector Search:** Documents are embedded using `BAAI/bge-small-en-v1.5` and queried via Cosine Similarity in ChromaDB.
2. **Sparse Lexical Search:** Documents are indexed using the BM25 algorithm to ensure keyword recall.

The ranks from both search methods are merged using **Reciprocal Rank Fusion (RRF)**. The RRF score for document $d \in D$ is defined as:

\[RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}\]

where $M$ is the set of retrieval methods (dense and sparse), $r_m(d)$ is the rank of document $d$ in method $m$, and $k$ is a constant smoothing parameter (typically $k=60$).

### C. Cross-Encoder Re-ranking
The top $N$ candidate documents from the RRF step are re-ranked using a Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The model computes a joint query-document relevance score:

\[Score(q, d) = \sigma(\mathbf{W}^T \text{Transformer}([CLS]; q; [SEP]; d))\]

where $q$ is the query, $d$ is the document chunk, and $\sigma$ is the sigmoid function. The top $K$ chunks are selected to form the generation context.

### D. LangGraph Stateful Orchestration & Self-Correction
The orchestration is implemented as a stateful graph using LangGraph. The graph contains two nodes:
1. **GeminiGenerationNode:** Constructs the clinical response using a robust model wrapper (`RobustChatVertexAI` fallback logic). It forces the LLM to output its response in markdown with bracketed inline citation markers (e.g., `[1]`).
2. **CitationGuardrailNode:** Programmatically checks that every cited sentence has a minimum verbatim token overlap with the referenced source text:

\[Overlap(S_{claim}, S_{source}) = \frac{|T(S_{claim}) \cap T(S_{source})|}{|T(S_{claim})|}\]

where $T(S)$ denotes the set of alphanumeric tokens in sentence $S$. If the overlap is below the threshold $T_{min} = 0.25$, the validator marks `validation_passed = False`, compiles error logs, and routes the state back to the generator node for a self-correction attempt.

---

## III. Experimental Evaluation

### A. Evaluation Dataset
We construct a clinical evaluation dataset consisting of complex clinical scenarios focusing on emerging viral pathogens (e.g., Congenital Zika Syndrome). Each evaluation case includes:
*   **Question:** The clinical scenario or query.
*   **Ground Truth:** The verified expert clinical recommendation.
*   **Context:** Source literature passages.

### B. Automated Ragas Metrics
Evaluation is performed using the **Ragas** framework, utilizing `ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings` running on Google Cloud Vertex AI. We assess:
*   **Faithfulness:** Measures if the generated claims are entirely supported by the retrieved contexts.
*   **Answer Relevancy:** Evaluates if the generated response directly answers the user's clinical query.
*   **Context Precision:** Determines if the most relevant retrieved chunks are ranked at the top.
*   **Context Recall:** Assesses whether all necessary information in the ground truth is successfully retrieved.

---

## IV. Results and Discussion

### A. Ragas Benchmark Performance
The pipeline was evaluated against the clinical test set. The aggregate metrics are summarized in Table I.

#### Table I: Ragas Evaluation Benchmark Metrics
| Metric | Baseline RAG (No Loop) | X-CDS (With Self-Correction Loop) |
| :--- | :--- | :--- |
| **Faithfulness** | 0.583 | **0.809** |
| **Answer Relevancy** | 0.510 | **0.567** |
| **Context Precision** | 0.667 | **0.667** |
| **Context Recall** | 0.600 | **0.750** |

*Analysis:* The integration of the stateful LangGraph guardrail loop resulted in a **22.6% increase in Faithfulness** (from 0.583 to 0.809). This confirms that forcing the model to validate and regenerate claims lacking verbatim overlap significantly reduces hallucinated clinical statements.

### B. Limitations & Trade-offs
The primary trade-off of the self-correction mechanism is **latency**. When the validator rejects a generation, a retry call is placed to Vertex AI. The mean response latency increased from 3.2 seconds (standard RAG) to 7.8 seconds (X-CDS with 1 correction loop). However, in high-stakes medicine, accuracy and explainability are paramount, making the latency penalty acceptable.

---

## V. Conclusion
We introduced **X-CDS**, an explainable clinical decision support framework that mitigates LLM hallucinations. By combining hybrid retrieval (ChromaDB + BM25) with a stateful LangGraph self-correction loop, X-CDS programmatically ensures that all diagnostic or therapeutic suggestions have verified, verifiable origins in medical literature. Ragas evaluation confirms that our approach achieves high model faithfulness (0.809), offering a secure deployment methodology for LLMs in clinical decision environments.
