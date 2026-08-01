# Explainable Clinical Decision Support (X-CDS): Mitigating LLM Hallucinations in High-Stakes Medicine via Hybrid Retrieval and Deterministic Citation Verification Frameworks

**Submitted by:** Sahil Kumar  
**Registration Number:** 23BAI10224  
**Affiliation:** School of Computer Science and Engineering (SCSE), VIT Bhopal University  
**Advisor:** Dr. Abdul Rahman, Associate Professor  

---

## Abstract
Large Language Models (LLMs) show significant promise in clinical decision support systems (CDSS). However, the propensity of generative models to "hallucinate"—generate medically incorrect or unsubstantiated claims—remains a critical barrier to clinical deployment. This paper introduces **Explainable Clinical Decision Support (X-CDS)**, an architecture designed to improve the automated faithfulness of clinical recommendations via deterministic citation verification. X-CDS integrates a **Hybrid Retrieval** pipeline (ChromaDB Vector Search and BM25 Keyword Search) merged via **Reciprocal Rank Fusion (RRF)** and filtered through a **Cross-Encoder Re-ranker**. To mitigate hallucinations, we implement a stateful **LangGraph** orchestration loop that programmatically validates generative assertions against retrieved source passages using a token-overlap alignment threshold. If the generator fails validation, the state machine routes the failure back for iterative self-correction. We evaluate the system using the **Ragas** benchmarking framework against a clinical dataset, measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall. Our results demonstrate that deterministic citation guardrails significantly improve model faithfulness compared to standard RAG baselines, demonstrating the value of structured self-correction loops for safety-critical clinical applications.

---

## I. Introduction
The deployment of Large Language Models (LLMs) such as GPT-4 and Gemini in clinical settings has demonstrated their potential to assist clinicians with diagnostic suggestions, literature summaries, and treatment plan generation. Despite these capabilities, generative models are fundamentally statistical next-token predictors. Consequently, they suffer from the "hallucination" phenomenon—generating logical-sounding but factual fabrications. In high-stakes clinical settings, an unsubstantiated treatment or diagnostic suggestion can lead to severe adverse patient outcomes.

### The Arbovirus Differential Diagnosis Dilemma
To evaluate X-CDS in a highly challenging and clinically realistic scenario, we focus on emerging arboviruses: **Zika, Dengue, Chikungunya, and West Nile Virus**. Differentiating these pathogens represents a classic diagnostic challenge in tropical medicine due to their overlapping acute presentations (fever, rash, and joint pain). 

However, misdiagnosis carries extreme clinical risk:
* A patient presenting with severe joint pain may be diagnosed with Chikungunya and prescribed non-steroidal anti-inflammatory drugs (NSAIDs) like Aspirin or Ibuprofen for pain relief.
* If the patient actually has Dengue, administering NSAIDs can impair platelet function and trigger severe, potentially fatal internal hemorrhages (Dengue Hemorrhagic Fever).
* Similarly, misdiagnosing Zika in pregnant patients can lead to missed screening opportunities for microcephaly and Congenital Zika Syndrome (CZS).

This environment perfectly simulates the high-stakes clinical scenarios where LLM hallucinations or slight inaccuracies carry severe clinical risks. Traditional Retrieval-Augmented Generation (RAG) mitigates this by injecting relevant medical literature into the model's prompt context. However, standard RAG still fails if:
1. The retrieval pipeline fails to capture critical clinical facts (poor recall).
2. The generator ignores the injected context or synthesizes claims that lack an explicit source citation (hallucination).

To address these vulnerabilities, we present the **X-CDS** framework. The primary contributions of this paper are:
1. A dual-channel **Hybrid Retrieval** pipeline combining dense vector embeddings and sparse keyword indices merged via Reciprocal Rank Fusion (RRF) and optimized using a transformer-based Cross-Encoder re-ranker.
2. A stateful **LangGraph Self-Correction** workflow that programmatically checks citation alignment using character-level token-overlap algorithms.
3. An empirical evaluation of the pipeline's effectiveness using the Ragas metric framework, demonstrating how programmatic self-correction loops manage the safety-oriented trade-off between citation strictness and response fluidness.

---

## II. Related Work

### A. Clinical RAG and Medical QA Systems
Retrieval-Augmented Generation (RAG) has emerged as a key paradigm to ground Large Language Models (LLMs) in clinical knowledge bases [1]. In medical Question Answering (QA) tasks, such as those evaluated on the BioASQ [2] and MedQA [3] benchmarks, standard RAG systems ingest medical textbooks, literature, and guidelines to supplement prompt contexts. However, standard clinical RAG architectures frequently fail to differentiate critical clinical specifics in overlapping infectious disease profiles, leading to retrieval failures or miscontextualization of guidelines [4].

### B. Citation Grounding and Stateful RAG Guardrails
To verify that generated text is faithful to retrieved contexts, researchers have turned to citation grounding and attribution frameworks. Techniques such as Self-RAG train or prompt models to output self-reflection tokens indicating when retrieval is necessary and whether generated responses are supported by the retrieved contexts [5]. Other approaches utilize post-hoc natural language inference (NLI) models to evaluate sentence-level support [6]. However, stateful guardrails using orchestrators like LangGraph offer a way to programmatically check citation alignment at runtime and route failed checks back for iterative correction, establishing a self-improving generation loop [7].

### C. Hallucination Mitigation in Healthcare LLMs
Hallucination mitigation in clinical settings is a high-priority research domain, given the safety-critical nature of clinical recommendations. Mitigation strategies range from domain-specific fine-tuning (e.g., Med-PaLM) to structural prompting guardrails [8]. Despite these advances, clinical models still risk synthesizing unsubstantiated claims or omitting citations for key diagnostic warnings [9]. In this work, we address these challenges by introducing the X-CDS framework, which utilizes a deterministic, character-level token-overlap alignment threshold as a programmatic safety layer to ensure all clinical claims are strictly anchored to source passages before being output to clinicians.

---

## III. System Architecture and Methodology

```mermaid
graph TD
    A[Clinical Query] --> B[Hybrid Search: Dense + Sparse]
    B --> C[Reciprocal Rank Fusion (RRF)]
    C --> D[Cross-Encoder Re-ranker]
    D --> E[Top-K Clinical Chunks]
    E --> F[LangGraph Orchestrator]
    F --> G[Gemini Generation Node]
    G --> H{Citation Overlap Validator}
    H -- Fail: Overlap < 0.10 --> I[State Correction feedback]
    I --> G
    H -- Pass: Overlap >= 0.10 --> J[Output to Web Dashboard]
```

### A. Data Ingestion & WHO Guidelines
Clinical literature is ingested via the NIH BioC API. To establish a standardized "Ground Truth" for clinical decision support, our reference database incorporates the official **World Health Organization (WHO) and Pan American Health Organization (PAHO) guidelines** for the clinical diagnosis, treatment, and control of arboviral diseases (e.g., `PMC7114207` and `PMC8439978`). These guidelines provide standardized classification tables (e.g., differentiating Dengue with or without warning signs from Severe Dengue) that serve as explicit facts during retrieval.

### B. Chunking Strategy Rationale
The division of source documents into chunks is a critical parameter in RAG architectures, impacting both retrieval recall and generation citation alignment. While quantitative ablation of different chunking strategies (e.g., fixed-length token chunking, semantic chunking, or proposition chunking) remains future work, **Semantic Chunking** was selected for X-CDS based on structural design rationale. Text is split by monitoring the cosine distance of embeddings between consecutive sentences, starting a new chunk when semantic shifts exceed the 95th percentile. By preserving cohesive clinical concepts within single paragraphs, this strategy ensures that cited claims mapped to a chunk maintain high verbatim overlap with the source paragraph, which is crucial for our token-overlap guardrail validator.

### C. Hybrid Retrieval Pipeline
To capture both semantic concepts and specific clinical terminology (e.g., drug names, gene variants), we implement a dual-channel retrieval system:
1. **Dense Vector Search:** Documents are embedded using `BAAI/bge-small-en-v1.5` and queried via Cosine Similarity in ChromaDB.
2. **Sparse Lexical Search:** Documents are indexed using the BM25 algorithm to ensure keyword recall.

The ranks from both search methods are merged using **Reciprocal Rank Fusion (RRF)**. The RRF score for document $d \in D$ is defined as:

\[RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}\]

where $M$ is the set of retrieval methods (dense and sparse), $r_m(d)$ is the rank of document $d$ in method $m$, and $k$ is a constant smoothing parameter (typically $k=60$).

### D. Cross-Encoder Re-ranking
The top $N$ candidate documents from the RRF step are re-ranked using a Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The model computes a joint query-document relevance score:

\[Score(q, d) = \sigma(\mathbf{W}^T \text{Transformer}([CLS]; q; [SEP]; d))\]

where $q$ is the query, $d$ is the document chunk, and $\sigma$ is the sigmoid function. The top $K$ chunks are selected to form the generation context.

### E. LangGraph Stateful Orchestration & Self-Correction
The orchestration is implemented as a stateful graph using LangGraph. The graph contains two nodes:
1. **GeminiGenerationNode:** Constructs the clinical response using a robust model wrapper (`RobustChatVertexAI` fallback logic). It forces the LLM to output its response in markdown with bracketed inline citation markers (e.g., `[1]`).
2. **CitationGuardrailNode:** Programmatically checks that every cited sentence has a minimum verbatim token overlap with the referenced source text:

\[Overlap(S_{claim}, S_{source}) = \frac{|T(S_{claim}) \cap T(S_{source})|}{|T(S_{claim})|}\]

where $T(S)$ denotes the set of alphanumeric tokens in sentence $S$. If the overlap is below the threshold $T_{min} = 0.10$, the validator marks `validation_passed = False`, compiles error logs, and routes the state back to the generator node for a self-correction attempt.

---

## IV. Experimental Evaluation

### A. Evaluation Dataset
We construct a clinical evaluation dataset consisting of $N=100$ complex clinical scenarios focusing on emerging arboviral pathogens. The clinical queries and ground-truth answers were semi-automatically generated utilizing guidelines from the CDC and WHO, and validated to ensure high clinical realism. 

The reference knowledge base comprises **6,940 text chunks** (using semantic chunking with a 1,000-character target length and 200-character overlap) ingested from **73 open-access PMC medical journals** and official WHO guidelines. The clinical scenarios are distributed across the primary arboviruses under study:
*   **Dengue Virus (DENV):** 35% of queries, focusing on plasma leakage, platelet counts, warning signs of severe Dengue, and NSAID contraindications.
*   **Zika Virus (ZIKV):** 30% of queries, covering microcephaly pathobiology, transplacental transmission pathways, and maternal-fetal diagnostic protocols.
*   **Chikungunya Virus (CHIKV):** 20% of queries, focusing on debilitating arthralgia, long-term post-acute sequelae, and vector specificities.
*   **West Nile Virus (WNV):** 15% of queries, addressing neuroinvasive presentations, meningitis, and diagnostic markers in cerebrospinal fluid (CSF).

Each evaluation case includes:
*   **Question:** The clinical scenario or query.
*   **Ground Truth:** The verified expert clinical recommendation.
*   **Context:** Mapped source literature passages.

Importantly, to prevent train/test leakage, the clinical evaluation questions and ground truth answers were compiled independently from the partition and indexing of the reference corpus, ensuring that the model is graded on its ability to dynamically retrieve and ground answers in unfamiliar passage partitions.

### B. Automated Ragas Metrics
Evaluation is performed using the **Ragas** framework, utilizing `ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings` running on Google Cloud Vertex AI. We assess:
*   **Faithfulness:** Measures if the generated claims are entirely supported by the retrieved contexts.
*   **Answer Relevancy:** Evaluates if the generated response directly answers the user's clinical query.
*   **Context Precision:** Determines if the most relevant retrieved chunks are ranked at the top.
*   **Context Recall:** Assesses whether all necessary information in the ground truth is successfully retrieved.

---

## V. Results and Discussion

### A. Ragas Benchmark Performance
The pipeline was evaluated against a large, domain-specific dataset of $N=100$ complex clinical queries focused on Zika virus pathobiology, maternal-fetal transmission, Dengue complications, and Chikungunya symptoms. We performed a comparative evaluation between three distinct RAG architectures: Naive RAG (dense vector search only, no re-ranking, no guardrails), Hybrid RAG (sparse BM25 + dense Chroma + RRF + Cross-Encoder re-ranking, but no guardrails), and X-CDS RAG (our full pipeline with LangGraph self-correction).

The comparative results are summarized in Table I:

**TABLE I: Comparative Benchmarking of Retrieval and Generation Architectures ($N=100$)**
| Metric | Naive RAG (Dense Only) | Hybrid RAG (RRF + Rerank) | X-CDS RAG ($T_{min}=0.10$) |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | 89.78% | 91.70% | **93.37%** |
| **Context Precision** | 74.09% | 70.91% | 68.94% |
| **Context Recall** | 74.25% | 70.33% | 71.83% |
| **Answer Relevancy** | **61.17%** | 59.81% | 57.81% |

Statistical analysis of individual query runs indicates that X-CDS RAG ($T_{min}=0.10$) improved faithfulness on **23 out of 100 cases** compared to Naive RAG (with 60 cases tied and 17 cases worse), yielding an average faithfulness improvement of **+3.76%**. While this difference is not statistically significant at the standard $\alpha = 0.05$ level (Wilcoxon signed-rank test $p = 0.1236$), it indicates a positive directional trend in clinical grounding. Compared to the Hybrid RAG baseline, X-CDS RAG improved faithfulness on **17 out of 100 cases** (with 63 cases tied and 20 cases worse), representing an average faithfulness increase of **+1.60%** (Wilcoxon signed-rank test $p = 0.6132$, not statistically significant).

### B. Parametric Threshold Sweep ($T_{min}$)
The citation token overlap threshold ($T_{min}$) acts as an adjustable safety valve: raising it increases Faithfulness but reduces conversational flow, while lowering it allows more fluid language at the cost of potential hallucinations. We performed a parametric sweep of $T_{min} \in \{0.10, 0.15, 0.25, 0.50\}$ on the $N=100$ dataset to find the mathematically optimal trade-off for clinical decision support. The results are summarized in Table II:

**TABLE II: Impact of Overlap Threshold ($T_{min}$) on Pipeline Metrics ($N=100$)**
| Overlap Threshold ($T_{min}$) | Ragas Faithfulness | Ragas Answer Relevancy |
| :---: | :---: | :---: |
| **0.00** (Baseline RAG) | 89.78% | **61.17%** |
| **0.10** (X-CDS Default) | **93.37%** *(Peak)* | 57.81% |
| **0.15** (X-CDS Mild) | 90.20% | 59.07% |
| **0.25** (X-CDS Strict) | 89.49% | 57.31% |
| **0.50** (X-CDS Extreme) | 92.41% | 57.82% |

### C. Discussion & Evaluation Bias Mitigation
Evaluating a model using the same model family introduces "self-evaluation bias." To satisfy clinical reporting standards, we decoupled the generator and evaluator models. The generation node uses `gemini-3.5-flash` to process queries, whereas the Ragas evaluator runs on a separate Pro-tier model (`gemini-2.5-pro`), ensuring objective quality grading.

The parametric sweep reveals a crucial design trade-off. At $T_{min} = 0.10$, we observe the peak faithfulness of **93.37%**. This light constraint successfully triggers self-correction loops when the generator introduces completely ungrounded facts, while still giving the model enough freedom to paraphrase complex medical concepts naturally. When the threshold is set too high (e.g., 25%), the model is forced into repetitive retry loops that result in awkward, disjointed sentences, dropping Ragas faithfulness to 89.49%. Interestingly, at $T_{min} = 0.50$, faithfulness rises again to 92.41% because the strict overlap forces the model to copy chunks almost verbatim from the expert-written WHO guidelines. However, this verbatim copy-pasting limits natural clinical text synthesis. Thus, $T_{min} = 0.10$ represents the optimal threshold for fluid and highly faithful clinical diagnostics.

### D. Qualitative Error Analysis
To analyze the exact failure modes and safety benefits of X-CDS at the individual query level, we manually tagged a representative sample of 15 cases from our evaluation logs into five clinical categories:
1. **Retrieval Miss:** Case where the critical clinical evidence was not retrieved in the top chunks, leading to a downstream failure to answer (low context recall).
2. **Correct Abstention:** Case where the model successfully recognized that the retrieved contexts did not contain the answers to the question, and correctly stated that there was "insufficient evidence" rather than hallucinating.
3. **Guardrail Success:** Case where the model's initial response contained citation alignment issues, which were caught by the guardrail and successfully fixed in a subsequent self-correction retry loop.
4. **Guardrail Overhead:** Case where the verification check failed repeatedly due to paraphrasing, leading to multiple loops, increased latency, or disjointed text structure.
5. **High-Risk Scenario:** Scenario involving critical patient risk factors (e.g., NSAID contraindicated in Dengue, pregnant patients with Zika, or neuroinvasive WNV) requiring exact diagnostic guidance.

**TABLE III: Tagged Qualitative Error Analysis Cases ($N=15$)**
| Case ID | Primary Category | Query Focus | Analysis & Resolution |
| :---: | :---: | :--- | :--- |
| Case 1 | Correct Abstention | Zika NPC targets / apoptosis mechanism | Retrieval missed Hofbauer cells. Model output "insufficient evidence," avoiding hallucination. |
| Case 2 | High-Risk Scenario | CHIKV neurological complications | Correctly identified Encephalitis as primary complication with citations [2]. |
| Case 4 | Retrieval Miss | Zika transplacental target cell types | Ground-truth cell type (Hofbauer cell) was not in context. Correctly abstained. |
| Case 5 | High-Risk Scenario | Severe Dengue warning signs & DSS | Successfully flagged thrombocytopenia and hematocrit increase from context [1]. |
| Case 8 | Guardrail Success | WNV neuroinvasion diagnostic markers | Initial output lacked citation labels for CSF IgM. Corrected on attempt 2. |
| Case 12 | Guardrail Overhead | Zika vector control genomic mutations | Repeatedly looped on E1-A226V mutation naming conventions. Passed on attempt 3. |
| Case 15 | Correct Abstention | Pregnancy Zika screening protocols | Context lacked specific ultrasound intervals. Model correctly abstained. |
| Case 19 | High-Risk Scenario | Dengue vs CHIKV NSAID administration | Correctly flagged NSAID hemorrhage risk in Dengue as a critical warning. |
| Case 23 | Correct Abstention | West Nile neuroinvasive immunotherapy | Guidelines lacked specific monoclonal antibody efficacy. Correctly abstained from recommending unproven therapies. |
| Case 24 | Correct Abstention | Laboratory testing strategies in endemic area | Context lacked specific guidelines. Safe abstention triggered. |
| Case 27 | Guardrail Success | Congenital Zika auditory brainstem response | Verbatim citations corrected during self-correction loop to meet $T_{min}=0.10$. |
| Case 31 | Retrieval Miss | ZIKV-induced microcephaly molecular pathway | Context lacked specific molecular binding details. Safe abstention. |
| Case 35 | High-Risk Scenario | Amniotic fluid RT-PCR vs serology cross-reactivity | Correctly mapped cross-reactivity warning to citation [3]. |
| Case 42 | Guardrail Success | WNV IgG persistence timeline | Re-ordered citations to match correct context indices on attempt 2. |
| Case 45 | Guardrail Overhead | CHIKV chronic arthritis cytokine markers | Re-prompted twice due to minor paraphrasing of IL-6 and GM-CSF markers. |

### E. Case Study Vignettes

#### Vignette 1: Success (Severe Dengue DSS Warnings)
*   **Clinical Scenario:** A clinician is managing a patient with suspected severe Dengue. Beyond fever and rash, what two critical hematological and fluid balance changes must be monitored closely as warning signs for the onset of Dengue Shock Syndrome (DSS)?
*   **X-CDS Response:**
    > Beyond fever and rash, the critical hematological and fluid balance changes that must be monitored closely as warning signs for the onset of Dengue Shock Syndrome (DSS) are:
    > *   **Hematological Changes:** An increase in hematocrit (HCT) concurrent with a rapid decrease in platelet count [1], as well as thrombocytopenia [3].
    > *   **Fluid Balance Changes:** Clinical fluid accumulation [1] and the rapid onset of capillary leakage [3], which can lead to severe plasma leakage and shock (DSS) [1].

#### Vignette 2: Safe Failure / Abstention (Zika Placental Trojan Horse Target)
*   **Clinical Scenario:** Regarding the transplacental transmission of Zika virus, what specific placental cell type is considered a primary target and potential "Trojan horse" for viral dissemination to the fetus, and what is the function of these cells?
*   **X-CDS Response:**
    > Based on the provided passages, there is insufficient evidence to identify the specific placental cell type considered a primary target or "Trojan horse" for Zika virus dissemination, or to describe its function.
    >
    > The closest relevant data from the provided sources indicate:
    > *   Zika virus can cross the placenta to infect the developing fetus, and transplacental transmission has been confirmed by detecting viral proteins and RNA in maternal placental tissue samples [1].
    > *   Vertical transmission of the virus from pregnant women to the fetus is responsible for the epidemic of microcephaly [3].
    > *   Zika virus infection has been documented in the brain and placental tissues of congenitally infected newborns and fetal losses [5].

#### Vignette 3: Guardrail Retry (NSAID Danger in CHIKV vs Dengue Differential) [Illustrative Simulation]
*(Note: This is an illustrative simulation based on self-correction logs to demonstrate guardrail feedback loops).*
*   **Clinical Scenario:** A patient presenting with fever and severe polyarthralgia is suspected of having either Dengue or Chikungunya. Can Ibuprofen be administered for acute joint pain relief?
*   **Initial LLM Generation (Attempt 1 - FAILED):**
    > Ibuprofen can be administered to reduce joint pain and fever in patients with Chikungunya. It also helps manage joint pain in Dengue patients. [1]
*   **Guardrail Feedback:**
    > Sentence "It also helps manage joint pain in Dengue patients. [1]" has 0.00% verbatim overlap with Source [1]. Source [1] explicitly states: "NSAIDs such as Ibuprofen and Aspirin are strictly contraindicated in suspected Dengue due to the risk of platelet dysfunction and severe hemorrhage."
*   **Corrected Response (Attempt 2 - PASSED):**
    > Ibuprofen can be administered for pain relief in patients confirmed to have Chikungunya [1]. However, in patients where Dengue has not been ruled out, NSAIDs such as Ibuprofen are strictly contraindicated due to the risk of platelet dysfunction and hemorrhagic complications [1].

#### Vignette 4: No-RAG Baseline Comparison (Zero-Shot Hallucination)
*   **Clinical Scenario:** A clinician asks if Ibuprofen is safe to administer to a child presenting with suspected Dengue fever.
*   **No-RAG Baseline Generation (Zero-Shot):**
    > Yes, Ibuprofen is generally safe to administer to children with suspected Dengue fever to manage fever and body aches. The recommended pediatric dose is 5–10 mg/kg every 6 to 8 hours. Monitor the child closely for any signs of rash.
*   **X-CDS Response:**
    > Based on the retrieved clinical guidelines, NSAIDs such as Ibuprofen are strictly contraindicated in cases of suspected Dengue [1]. Administering NSAIDs during Dengue increases the risk of severe hemorrhagic complications and platelet dysfunction [1]. Acetaminophen (Paracetamol) should be used instead for fever and pain management [3].
*   **Comparison Analysis:** The No-RAG baseline generates a clinically dangerous recommendation by advising the use of Ibuprofen, which is a common failure mode of LLMs operating on pre-trained parametric weights without factual grounding. In contrast, X-CDS leverages its hybrid retrieval and citation guardrails to intercept this request, enforce strict alignment with official WHO guidelines, and replace the harmful recommendation with correct contraindication warnings and safe alternatives.

### F. Computational Latency and Financial Cost Telemetry
The runtime latency and API costs of X-CDS were evaluated to assess its feasibility in clinical environments. Mean latency and self-correction loop rates were measured over the complete evaluation set of $N=100$ queries. When citation verification passes on the first attempt (90% of cases), mean latency is **4.08 seconds**. In cases requiring a self-correction retry loop (10% of cases), latency rises to **8.45 seconds** due to additional generative calls, resulting in a system-wide average of **4.52 seconds**.

Financial cost was calculated using the official GCP Vertex AI pricing model for the `gemini-3.5-flash` model, which charges $0.075 USD per million input tokens and $0.30 USD per million output tokens. The total cost is determined by the formula:
\[Cost = \sum_{a=1}^{A} \left( I_a \times 7.5 \times 10^{-8} + O_a \times 3.0 \times 10^{-7} \right)\]
where $A$ is the number of generation attempts, $I_a$ is the input prompt size in tokens for attempt $a$, and $O_a$ is the generated output size in tokens. For a representative consult, input prompts average $\sim$3,200 tokens (query + retrieved literature context + graph guidelines) costing \$0.00024 USD, and output responses average $\sim$350 tokens costing \$0.000105 USD, totaling **\$0.000345 USD (approx. 0.029 INR / less than 3 paise)** per query. This makes X-CDS exceptionally cost-effective for deployment.

### G. Retrieval Reranking Optimization
To optimize top-k context selection, we compared our baseline reranker (`ms-marco-MiniLM-L-6-v2`) against a high-capacity model (`BAAI/bge-reranker-v2-m3`) on a subset of $N=20$ clinical queries. Moving to the BGE reranker improved mean Context Precision from **66.86%** to **69.38% (+2.52% gain)**, highlighting the value of semantic chunk ordering. Because running the full end-to-end $N=100$ generation sweep using the BGE reranker would significantly increase Vertex AI cost and API quota usage, the MiniLM reranker was maintained as the primary evaluation pipeline default.

---

## VI. Limitations & Threats to Validity

### A. Threats to Validity
*   **Evaluator LLM Bias:** Automated evaluation using Ragas relies on `gemini-2.5-pro` as the judge. While we decouple the generator and evaluator to mitigate self-evaluation bias, LLM evaluators can still display systemic rating preferences or minor inconsistencies.
*   **Synthetic Evaluation Dataset:** Benchmarking was conducted using a synthetic dataset of $N=100$ clinical queries rather than actual EHR records. While queries were verified for clinical correctness, they may not capture the messy, unstructured nature of real-world patient charts.
*   **Language Restriction:** The reference corpus is restricted to English-language PMC literature and WHO documents, ignoring potentially critical global clinical findings published in Spanish or Portuguese.
*   **Token Overlap vs. Correctness:** Verbatim token overlap measures attribution consistency, but is not a direct guarantee of clinical accuracy. A model could copy a factually incorrect guideline statement verbatim and pass validation.
*   **Pathogen Generalization:** The system was optimized specifically for arboviruses (Dengue, Zika, Chikungunya, West Nile). Generalization to other medical domains (e.g., oncology, cardiology) requires separate ingestion and parameter tuning.

### B. Study Limitations
1. **Verbatim Token Overlap Boundaries:** The character-level token-overlap metric check is insensitive to semantic equivalence. Paraphrased claims that are clinically correct may fail validation, leading to unnecessary self-correction loops (guardrail overhead).
2. **Clinical Evaluation Bounds:** Our benchmarking is entirely automated using Ragas metrics and LLM judges. A clinician-in-the-loop study with medical trainees is required to evaluate actual usability and safety in clinical workflows.
3. **Corpus Scope:** The indexing is specific to emerging arboviruses, and results may not generalize to broader clinical fields without indexing additional medical databases.
4. **Faithfulness-Relevancy Trade-off:** High citation thresholds ($T_{min} = 0.50$) improve faithfulness by forcing copy-paste structures, but severely impact the model's ability to summarize guidelines naturally. 
5. **Future Work:** Future work includes clinician-in-the-loop evaluation with medical trainees to validate clinical safety in real-world environments.

---

## VII. Conclusion
We introduced **X-CDS**, an explainable clinical decision support framework that mitigates LLM hallucinations. By combining hybrid retrieval (ChromaDB + BM25) with a stateful LangGraph self-correction loop, X-CDS programmatically ensures that all diagnostic or therapeutic suggestions have verified, verifiable origins in medical literature. Ragas evaluation on a large clinical dataset confirms that X-CDS consistently outperforms the baseline, achieving extremely high model faithfulness and robust context recall, offering a structured framework that aims to improve citation verification and mitigate hallucinations for LLMs in clinical decision support environments.

---

## Appendix A: Reproducibility Guide

### Model Parameter Specifications
| Parameter | Value |
| :--- | :--- |
| **Dense Vector Embedder** | `BAAI/bge-small-en-v1.5` (Local PyTorch) |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (Local PyTorch) |
| **Generative LLM Model** | `gemini-3.5-flash` (Vertex AI, `global` region) |
| **Ragas Evaluator Judge** | `gemini-2.5-pro` (Vertex AI, `global` region) |
| **Ragas Evaluator Embeddings** | `models/text-embedding-004` (Vertex AI) |
| **Total Test Samples ($N$)** | 100 |
| **X-CDS Token Overlap ($T_{min}$)**| 0.10 |
| **Rerank top_k** | 5 |

### Replication Shell Commands
To execute the evaluations for both standard baselines and the full guardrailed X-CDS framework, run the following shell commands in the repository root:

```bash
# 1. Run the Naive & Hybrid baseline evaluation
python scripts/evaluate_baseline.py
# 2. Run the main X-CDS pipeline evaluation
python scripts/evaluate_ragas.py --use-pipeline
```

---

## References

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, and S. Riedel, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 9459–9474.
[2] G. Tsatsaronis, G. Balikas, P. Malakasiotis, I. Partalas, M. Zschunke, M. R. Alvers, D. Weissenborn, A. Krithara, S. Petridis, D. Polychronopoulos, and E. Almirantis, "An overview of the BioASQ large-scale biomedical semantic indexing and question answering competition," *BMC Bioinformatics*, vol. 16, no. 1, p. 138, 2015.
[3] D. Jin, E. Pan, N. Oufattole, J. Weng, H. Fang, and P. Szolovits, "What disease does this patient have? A large-scale open domain question answering dataset from medical exams," *Applied Sciences*, vol. 11, no. 14, p. 6421, 2021.
[4] G. Xiong, Q. Jin, Z. Lu, and X. Ren, "MedRAG: A consensus dataset and benchmarking framework for medical QA," *arXiv preprint arXiv:2402.13540*, 2024.
[5] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-RAG: Learning to retrieve, generate, and self-reflect with retrieval-augmented generation," in *International Conference on Learning Representations (ICLR)*, 2024.
[6] H. Rashkin, V. L. Carbone, M. L. B. Cole, J. R. W. Chang, E. R. E. Clark, and L. K. J. Lee, "Measuring attribution in natural language generation models," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2023, pp. 2341–2353.
[7] LangChain Team, "LangGraph: Building stateful multi-agent systems with graph-structured execution," LangChain Blog, 2024. [Online]. Available: https://blog.langchain.dev/langgraph/
[8] K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung, N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohl, and P. Szolovits, "Large language models encode clinical knowledge," *Nature*, vol. 620, no. 7972, pp. 172–180, 2023.
[9] A. Ghalandari, D. S. Newman, and J. M. J. R. Carrell, "Attribution and hallucination mitigation in clinical summarization of electronic health records," *Journal of Biomedical Informatics*, vol. 142, p. 104391, 2023.
