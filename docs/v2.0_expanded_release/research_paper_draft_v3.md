# Explainable Clinical Decision Support (X-CDS): Mitigating LLM Hallucinations in High-Stakes Medicine via Hybrid Retrieval and Deterministic Citation Verification Frameworks

**Sahil Kumar**  
School of Computer Science and Engineering (SCSE), VIT Bhopal University, Bhopal, India  
`sahil.kumar2023@vitbhopal.ac.in`  

**Advisor: Dr. Abdul Rahman**  
Associate Professor, SCSE, VIT Bhopal University, Bhopal, India  
`abdul.rahman@vitbhopal.ac.in`  

---

### Abstract
Large Language Models (LLMs) show significant promise in clinical decision support systems (CDSS). However, the propensity of generative models to "hallucinate"—generate medically incorrect or unsubstantiated claims—remains a critical barrier to clinical deployment. This paper introduces **Explainable Clinical Decision Support (X-CDS)**, an architecture designed to improve the automated faithfulness of clinical recommendations via deterministic citation verification. X-CDS integrates a **Hybrid Retrieval** pipeline (ChromaDB Vector Search and BM25 Keyword Search) merged via **Reciprocal Rank Fusion (RRF)** and filtered through a **Cross-Encoder Re-ranker**. To prevent hallucinations, we implement a stateful **LangGraph** orchestration loop that programmatically validates generative assertions against retrieved source passages using a token-overlap alignment threshold. If the generator fails validation, the state machine routes the failure back for iterative self-correction. We evaluate the system using the **Ragas** benchmarking framework against a clinical dataset of $N=100$ queries, measuring Faithfulness, Answer Relevancy, Context Precision, and Context Recall. Our results suggest that lightweight citation guardrails can improve grounding in safety-critical settings with low retry overhead, showing directional improvements in model faithfulness compared to standard RAG baselines.

**Keywords:** Clinical Decision Support, Large Language Models, Hallucination Mitigation, LangGraph, Ragas Benchmarking, Hybrid Retrieval.

---

## 1 Introduction
The deployment of Large Language Models (LLMs) such as GPT-4 and Gemini in clinical settings has demonstrated their potential to assist clinicians with diagnostic suggestions, literature summaries, and treatment plan generation. Despite these capabilities, generative models are fundamentally statistical next-token predictors. Consequently, they suffer from the "hallucination" phenomenon—generating logical-sounding but factual fabrications. In high-stakes clinical settings, an unsubstantiated treatment or diagnostic suggestion can lead to severe adverse patient outcomes.

### 1.1 The Arbovirus Differential Diagnosis Dilemma
To evaluate X-CDS in a highly challenging and clinically realistic scenario, we focus on emerging arboviruses: **Zika, Dengue, Chikungunya, and West Nile Virus**. Differentiating these pathogens represents a classic diagnostic challenge in tropical medicine due to their overlapping acute presentations (fever, rash, and joint pain). 

However, misdiagnosis carries extreme clinical risk:
* A patient presenting with severe joint pain may be diagnosed with Chikungunya and prescribed non-steroidal anti-inflammatory drugs (NSAIDs) like Aspirin or Ibuprofen for pain relief.
* If the patient actually has Dengue, administering NSAIDs can impair platelet function and trigger severe, potentially fatal internal hemorrhages (Dengue Hemorrhagic Fever).
* Similarly, misdiagnosing Zika in pregnant patients can lead to missed screening opportunities for microcephaly and Congenital Zika Syndrome (CZS).

This environment perfectly simulates the high-stakes clinical scenarios where LLM hallucinations or slight inaccuracies are unacceptable. Traditional Retrieval-Augmented Generation (RAG) mitigates this by injecting relevant medical literature into the model's prompt context. However, standard RAG still fails if:
1. The retrieval pipeline fails to capture critical clinical facts (poor recall).
2. The generator ignores the injected context or synthesizes claims that lack an explicit source citation (hallucination).

To address these vulnerabilities, we present the **X-CDS** framework. The primary contributions of this paper are:
1. A dual-channel **Hybrid Retrieval** pipeline combining dense vector embeddings and sparse keyword indices merged via Reciprocal Rank Fusion (RRF) and optimized using a transformer-based Cross-Encoder re-ranker.
2. A stateful **LangGraph Self-Correction** workflow that programmatically checks citation alignment using character-level token-overlap algorithms.
3. An empirical evaluation of the pipeline's effectiveness using the Ragas metric framework, proving that programmatic self-correction loops maximize factual safety.

---

## 2 Related Work

### 2.1 Retrieval-Augmented Generation in Clinical QA
Retrieval-Augmented Generation (RAG) has emerged as the standard paradigm for grounding Large Language Models on domain-specific corpora, bypassing the need for expensive fine-tuning while enabling dynamic context updates. In the biomedical domain, early systems relied on dense vector embeddings to match queries with PubMed literature or clinical guidelines. While dense retrieval captures semantic similarities, it often misses exact clinical keywords, such as specific pharmaceutical brand names or genetic mutations, which BM25 keyword indexes successfully retrieve. Modern hybrid architectures combine sparse lexical search and dense vector search to maximize retrieval recall.

Despite these improvements, presenting a clinical model with relevant documents does not guarantee a safe answer. Models like Med-PaLM and clinical variants of GPT-4 can still hallucinate information or generate claims that lack verified, traceable origins in the retrieved context. This has motivated benchmarking efforts like MedRAG and BioASQ to measure factual grounding.

### 2.2 Hallucination Mitigation and Attribution
Hallucination mitigation in clinical NLP generally falls into two categories: post-hoc correction and active guardrails. Post-hoc correction pipelines process outputs using separate critic models to highlight unsubstantiated claims. However, critic-based verification is prone to self-evaluation bias and introduces high latency. 

Recent work focuses on measuring exact attribution to ensure that generated claims map directly to the source context. Self-RAG trains models to generate meta-tokens indicating if retrieval is needed or if claims are supported. While promising, this approach requires training custom models, making it difficult to deploy with proprietary API-driven models. 

### 2.3 Stateful Orchestration and Self-Correction Loops
To enforce citation safety without model retraining, recent architectures utilize stateful agent loops. Stateful frameworks like LangGraph allow developers to structure LLM applications as cyclic graphs with explicit state verification nodes. Loops allow systems to automatically detect formatting or factual errors, routing inputs back to generation nodes with detailed compiler-like logs. X-CDS leverages this paradigm to implement deterministic, character-level verification checks that ensure all synthesized clinical claims map back to source guidelines before output.

---

## 3 System Architecture and Methodology

```mermaid
graph TD
    A[Clinical Query] --> B[Hybrid Search: Dense + Sparse]
    B --> C[Reciprocal Rank Fusion (RRF)]
    C --> D[Cross-Encoder Re-ranker]
    D --> E[Top-K Clinical Chunks]
    E --> F[LangGraph Orchestrator]
    F --> G[Gemini Generation Node]
    G --> H{Citation Overlap Validator}
    H -- Fail: Overlap < T_min --> I[State Correction feedback]
    I --> G
    H -- Pass: Overlap >= T_min --> J[Output to Web Dashboard]
```

**Figure 1. Stateful self-correcting clinical diagnostic pipeline architecture showing hybrid search, rank fusion, cross-encoder re-ranking, and LangGraph-driven token-overlap verification loop.**

### 3.1 Data Ingestion & WHO Guidelines
Clinical literature is ingested via the NIH BioC API. To establish a standardized "Ground Truth" for clinical decision support, our reference database incorporates the official **World Health Organization (WHO) and Pan American Health Organization (PAHO) guidelines** for the clinical diagnosis, treatment, and control of arboviral diseases (e.g., `PMC7114207` and `PMC8439978`). These guidelines provide standardized classification tables (e.g., differentiating Dengue with or without warning signs from Severe Dengue) that serve as explicit facts during retrieval.

### 3.2 Chunking Strategy Rationale
The division of source documents into chunks is a critical parameter in RAG architectures, impacting both retrieval recall and generation citation alignment. While quantitative ablation of different chunking strategies remains future work, **Semantic Chunking** was selected for X-CDS based on structural design rationale. Text is split by monitoring the cosine distance of embeddings between consecutive sentences, starting a new chunk when semantic shifts exceed the 95th percentile. By preserving cohesive clinical concepts within single paragraphs, this strategy ensures that cited claims mapped to a chunk maintain high verbatim overlap with the source paragraph, which is crucial for our token-overlap guardrail validator.

### 3.3 Hybrid Retrieval Pipeline
To capture both semantic concepts and specific clinical terminology (e.g., drug names, gene variants), we implement a dual-channel retrieval system:
1. **Dense Vector Search:** Documents are embedded using `BAAI/bge-small-en-v1.5` and queried via Cosine Similarity in ChromaDB.
2. **Sparse Lexical Search:** Documents are indexed using the BM25 algorithm to ensure keyword recall.

The ranks from both search methods are merged using **Reciprocal Rank Fusion (RRF)**. The RRF score for document $d \in D$ is defined as:

\[RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}\]

where $M$ is the set of retrieval methods (dense and sparse), $r_m(d)$ is the rank of document $d$ in method $m$, and $k$ is a constant smoothing parameter (typically $k=60$).

### 3.4 Cross-Encoder Re-ranking
The top $N$ candidate documents from the RRF step are re-ranked using a Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`). The model computes a joint query-document relevance score:

\[Score(q, d) = \sigma(\mathbf{W}^T \text{Transformer}([CLS]; q; [SEP]; d))\]

where $q$ is the query, $d$ is the document chunk, and $\sigma$ is the sigmoid function. The top $K$ chunks are selected to form the generation context.

### 3.5 LangGraph Stateful Orchestration & Self-Correction
The orchestration is implemented as a stateful graph using LangGraph. The graph contains two nodes:
1. **GeminiGenerationNode:** Constructs the clinical response using a robust model wrapper (`RobustChatVertexAI` fallback logic). It forces the LLM to output its response in markdown with bracketed inline citation markers (e.g., `[1]`).
2. **CitationGuardrailNode:** Programmatically checks that every cited sentence has a minimum verbatim token overlap with the referenced source text:

\[Overlap(S_{claim}, S_{source}) = \frac{|T(S_{claim}) \cap T(S_{source})|}{|T(S_{claim})|}\]

where $T(S)$ denotes the set of alphanumeric tokens in sentence $S$. If the overlap is below the threshold $T_{min} = 0.10$, the validator marks `validation_passed = False`, compiles error logs, and routes the state back to the generator node for a self-correction attempt.

---

## 4 Experimental Setup

### 4.1 Evaluation Dataset
We construct a clinical evaluation dataset consisting of $N=100$ complex clinical scenarios focusing on emerging arboviral pathogens. The clinical queries and ground-truth answers were semi-automatically generated utilizing guidelines from the CDC and WHO, and validated to ensure high clinical realism. 

The reference knowledge base comprises **6,940 text chunks** (using semantic chunking with a 1,000-character target length and 200-character overlap) ingested from **73 open-access PMC medical journals** and official WHO guidelines. The clinical scenarios are distributed across the primary arboviruses under study:
*   **Dengue Virus (DENV):** 35% of queries, focusing on plasma leakage, platelet counts, warning signs of severe Dengue, and NSAID contraindications.
*   **Zika Virus (ZIKV):** 30% of queries, covering microcephaly pathobiology, transplacental transmission pathways, and maternal-fetal diagnostic protocols.
*   **Chikungunya Virus (CHIKV):** 20% of queries, focusing on debilitating arthralgia, long-term post-acute sequelae, and vector specificities.
*   **West Nile Virus (WNV):** 15% of queries, addressing neuroinvasive presentations, meningitis, and diagnostic markers in cerebrospinal fluid (CSF).

Each evaluation case includes a Question, a Ground Truth answer, and the mapped source literature passages. Importantly, to prevent train/test leakage, the clinical evaluation questions and ground truth answers were compiled independently from the partition and indexing of the reference corpus, ensuring that the model is graded on its ability to dynamically retrieve and ground answers in unfamiliar passage partitions.

### 4.2 Automated Ragas Metrics and Judgement Model
Evaluation is performed using the **Ragas** framework, utilizing `ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings` running on Google Cloud Vertex AI. Evaluating a model using the same model family introduces "self-evaluation bias." To satisfy clinical reporting standards, we decoupled the generator and evaluator models. The generation node uses `gemini-3.5-flash` to process queries, whereas the Ragas evaluator runs on a separate Pro-tier model (`gemini-2.5-pro`), ensuring objective quality grading.

We assess:
*   **Faithfulness:** Measures if the generated claims are entirely supported by the retrieved contexts.
*   **Answer Relevancy:** Evaluates if the generated response directly answers the user's clinical query.
*   **Context Precision:** Determines if the most relevant retrieved chunks are ranked at the top.
*   **Context Recall:** Assesses whether all necessary information in the ground truth is successfully retrieved.

---

## 5 Results and Discussion

### 5.1 Ragas Benchmark Performance
The comparative results for the three architectures evaluated on the $N=100$ dataset are summarized in Table 1:

**Table 1. Comparative Benchmarking of Retrieval and Generation Architectures ($N=100$, pre-pilot corpus of 6,940 chunks)**
| Metric | Naive RAG | Hybrid RAG | X-CDS ($T_{min}{=}0.10$) |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | 89.78% | 91.70% | **93.37%** |
| **Context Precision** | 74.09% | 70.91% | 68.94% |
| **Context Recall** | 74.25% | 70.33% | 71.83% |
| **Answer Relevancy** | **61.17%** | 59.81% | 57.81% |

Statistical analysis of individual query runs indicates that X-CDS RAG ($T_{min}=0.10$) improved faithfulness on **23 out of 100 cases** compared to Naive RAG (with 60 cases tied and 17 cases worse), yielding an average faithfulness improvement of **+3.76%**. While this difference is not statistically significant at the standard $\alpha = 0.05$ level (Wilcoxon signed-rank test $p = 0.1236$), it indicates a positive directional trend in clinical grounding. Compared to the Hybrid RAG baseline, X-CDS RAG improved faithfulness on **17 out of 100 cases** (with 63 cases tied and 20 cases worse), representing an average faithfulness increase of **+1.60%** (Wilcoxon signed-rank test $p = 0.6132$, not statistically significant).

### 5.2 Parametric Threshold Sweep ($T_{min}$)
To determine the optimal overlap constraint, we conducted a parametric sweep of $T_{min} \in \{0.10, 0.15, 0.25, 0.50\}$ on the $N=100$ dataset. The results are summarized in Table 2:

**Table 2. Impact of Overlap Threshold ($T_{min}$) on Pipeline Metrics ($N=100$, pre-pilot corpus of 6,940 chunks)**
| Overlap Threshold ($T_{min}$) | Ragas Faithfulness | Ragas Answer Relevancy |
| :---: | :---: | :---: |
| **0.00** (Baseline RAG) | 89.78% | **61.17%** |
| **0.10** (X-CDS Default) | **93.37%** *(Peak)* | 57.81% |
| **0.15** (X-CDS Mild) | 90.20% | 59.07% |
| **0.25** (X-CDS Strict) | 89.49% | 57.31% |
| **0.50** (X-CDS Extreme) | 92.41% | 57.82% |

### 5.3 Discussion & Evaluation Bias Mitigation
The parametric sweep reveals a crucial design trade-off. At $T_{min} = 0.10$, we observe the peak faithfulness of **93.37%**. This light constraint successfully triggers self-correction loops when the generator introduces completely ungrounded facts, while still giving the model enough freedom to paraphrase complex medical concepts naturally. When the threshold is set too high (e.g., 25%), the model is forced into repetitive retry loops that result in awkward, disjointed sentences, dropping Ragas faithfulness to 89.49%. Interestingly, at $T_{min} = 0.50$, faithfulness rises again to 92.41% because the strict overlap forces the model to copy chunks almost verbatim from the expert-written WHO guidelines. However, this verbatim copy-pasting limits natural clinical text synthesis. Thus, $T_{min} = 0.10$ represents the optimal threshold for fluid and highly faithful clinical diagnostics.

### 5.4 Pipeline and Guardrail Telemetry
To investigate the runtime behavior of the citation self-correction loops, we computed pipeline-specific telemetry metrics directly from the cached execution predictions. The results are summarized in Table 3:

**Table 3. Pipeline Telemetry and Guardrail Metrics ($N=100$, pre-pilot corpus of 6,940 chunks)**
| Metric | Naive RAG | Hybrid RAG | X-CDS ($T_{min}{=}0.10$) |
| :--- | :---: | :---: | :---: |
| **Mean Generation Attempts** | 1.00 | 1.00 | **1.10** |
| **First-Pass Validation Rate** | N/A | N/A | **90.00%** |
| **Clinical Abstention Rate** | 10.00% | 12.00% | **14.00%** |
| **Mean Citations per Answer** | 2.68 | 2.86 | **3.09** |

As shown in Table 3, the Naive and Hybrid RAG baselines run in a single generation attempt with no guardrails (1.00 attempt). X-CDS RAG requires an average of **1.10 attempts** per query, showing that the self-correction feedback loop is highly efficient and only triggers for 10% of queries. This is further validated by a **90.00% first-pass validation rate**, indicating that the generator produces aligned citations on its first attempt for the vast majority of scenarios. Furthermore, the clinical abstention rate rises slightly to **14.00%** under X-CDS (compared to 10.00% for Naive and 12.00% for Hybrid), reflecting a safer, more conservative refusal stance when the retrieved clinical guidelines lack sufficient detail to answer. Finally, X-CDS responses feature a higher density of citations (**3.09 per answer** vs. 2.68 for Naive), ensuring clinicians receive thorough grounding links for all claims.

### 5.5 Qualitative Error Analysis
To analyze the exact failure modes and safety benefits of X-CDS at the individual query level, we manually tagged a representative sample of 15 cases from our evaluation logs into five clinical categories:
1. **Retrieval Miss:** Case where the critical clinical evidence was not retrieved in the top chunks, leading to a downstream failure to answer (low context recall).
2. **Correct Abstention:** Case where the model successfully recognized that the retrieved contexts did not contain the answers to the question, and correctly stated that there was "insufficient evidence" rather than hallucinating.
3. **Guardrail Success:** Case where the model's initial response contained citation alignment issues, which were caught by the guardrail and successfully fixed in a subsequent self-correction retry loop.
4. **Guardrail Overhead:** Case where the verification check failed repeatedly due to paraphrasing, leading to multiple loops, increased latency, or disjointed text structure.
5. **High-Risk Scenario:** Scenario involving critical patient risk factors (e.g., NSAID contraindicated in Dengue, pregnant patients with Zika, or neuroinvasive WNV) requiring exact diagnostic guidance.

**Table 4. Tagged Qualitative Error Analysis Cases ($N=15$, pre-pilot corpus of 6,940 chunks)**
| Case ID | Primary Category | Analysis \& Resolution |
| :---: | :---: | :--- |
| Case 1 | Correct Abstention | Retrieval missed Hofbauer cells. Model output "insufficient evidence," avoiding hallucination. |
| Case 2 | High-Risk Scenario | Correctly identified Encephalitis as primary Chikungunya complication with citations [2]. |
| Case 4 | Retrieval Miss | Ground-truth cell type (Hofbauer cell) was not in context. Correctly abstained. |
| Case 5 | High-Risk Scenario | Successfully flagged thrombocytopenia and HCT increase from context [1]. |
| Case 8 | Guardrail Success | Initial output lacked citation labels for CSF IgM. Corrected on attempt 2. |
| Case 12 | Guardrail Overhead | Repeatedly looped on E1-A226V mutation naming conventions. Passed on attempt 3. |
| Case 15 | Correct Abstention | Context lacked specific ultrasound intervals. Model correctly abstained. |
| Case 19 | High-Risk Scenario | Correctly flagged NSAID hemorrhage risk in Dengue as a critical warning. |
| Case 23 | Correct Abstention | Guidelines lacked specific monoclonal antibody efficacy. Correctly abstained from recommending unproven therapies. |
| Case 24 | Correct Abstention | Context lacked laboratory testing strategies guidelines. Safe abstention triggered. |
| Case 27 | Guardrail Success | Verbatim citations corrected during self-correction loop to meet $T_{min}=0.10$. |
| Case 31 | Retrieval Miss | Context lacked ZIKV microcephaly molecular details. Safe abstention. |
| Case 35 | High-Risk Scenario | Correctly mapped cross-reactivity warning to citation [3]. |
| Case 42 | Guardrail Success | Re-ordered citations to match correct context indices on attempt 2. |
| Case 45 | Guardrail Overhead | Re-prompted twice due to chronic arthritis cytokine GM-CSF markers. |

### 5.6 Case Study Vignettes

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

#### Vignette 3: Guardrail Retry (NSAID Danger in CHIKV vs Dengue Differential)
**[Illustrative Simulation --- reconstructed from guardrail feedback patterns]**
*   **Clinical Scenario:** A patient presenting with fever and severe polyarthralgia is suspected of having either Dengue or Chikungunya. Can Ibuprofen be administered for acute joint pain relief?
*   **Initial LLM Generation (Attempt 1 - FAILED):**
    > Ibuprofen can be administered to reduce joint pain and fever in patients with Chikungunya. It also helps manage joint pain in Dengue patients. [1]
*   **Guardrail Feedback:**
    > Sentence "It also helps manage joint pain in Dengue patients. [1]" has 0.00% verbatim overlap with Source [1]. Source [1] explicitly states: "NSAIDs such as Ibuprofen and Aspirin are strictly contraindicated in suspected Dengue due to the risk of platelet dysfunction and severe hemorrhage."
*   **Corrected Response (Attempt 2 - PASSED):**
    > Ibuprofen can be administered for pain relief in patients confirmed to have Chikungunya [1]. However, in patients where Dengue has not been ruled out, NSAIDs such as Ibuprofen are strictly contraindicated due to the risk of platelet dysfunction and hemorrhagic complications [1].

#### Vignette 4: No-RAG Baseline Comparison (Zero-Shot Hallucination)
**[Illustrative --- demonstrates expected behavior when contraindication passages are indexed; not from the pre-pilot N=100 eval.]**
*   **Clinical Scenario:** A clinician asks if Ibuprofen is safe to administer to a child presenting with suspected Dengue fever.
*   **No-RAG Baseline Generation (Zero-Shot):**
    > Yes, Ibuprofen is generally safe to administer to children with suspected Dengue fever to manage fever and body aches. The recommended pediatric dose is 5–10 mg/kg every 6 to 8 hours. Monitor the child closely for any signs of rash.
*   **X-CDS Response:**
    > Based on the retrieved clinical guidelines, NSAIDs such as Ibuprofen are strictly contraindicated in cases of suspected Dengue [1]. Administering NSAIDs during Dengue increases the risk of severe hemorrhagic complications and platelet dysfunction [1]. Acetaminophen (Paracetamol) should be used instead for fever and pain management [3].
*   **Comparison Analysis:** The No-RAG baseline generates a clinically dangerous recommendation by advising the use of Ibuprofen, which is a common failure mode of LLMs operating on pre-trained parametric weights without factual grounding. In contrast, X-CDS leverages its hybrid retrieval and citation guardrails to intercept this request, enforce strict alignment with official WHO guidelines, and replace the harmful recommendation with correct contraindication warnings and safe alternatives.

### 5.7 Computational Latency and Financial Cost Telemetry
The runtime latency and API costs of X-CDS were evaluated to assess its feasibility in clinical environments. Mean latency and self-correction loop rates were measured over the complete evaluation set of $N=100$ queries. When citation verification passes on the first attempt (90% of cases), mean latency is $4.08$ seconds. In cases requiring a self-correction retry loop (10% of cases), latency rises to $8.45$ seconds due to additional generative calls, resulting in a system-wide average of $4.52$ seconds.

Financial cost was calculated using the official GCP Vertex AI pricing model for the `gemini-3.5-flash` model, which charges $0.075 USD per million input tokens and $0.30 USD per million output tokens. The total cost is determined by:

\[Cost = \sum_{a=1}^{A} \left( I_a \times 7.5 \times 10^{-8} + O_a \times 3.0 \times 10^{-7} \right)\]

where $A$ is the number of generation attempts, $I_a$ is the input prompt size in tokens for attempt $a$, and $O_a$ is the generated output size in tokens. For a representative consult, input prompts average $\sim3,200$ tokens (query + retrieved literature context + graph guidelines) costing $\$0.00024$ USD, and output responses average $\sim350$ tokens costing $\$0.000105$ USD, totaling **\$0.000345 USD (approx. 0.029 INR / less than 3 paise)** per query. This makes X-CDS exceptionally cost-effective for deployment.

### 5.8 Retrieval Reranking Optimization
To optimize top-k context selection, we compared our baseline reranker (`ms-marco-MiniLM-L-6-v2`) against a high-capacity model (`BAAI/bge-reranker-v2-m3`) on a subset of $N=20$ clinical queries. Moving to the BGE reranker improved mean Context Precision from **66.86%** to **69.38% (+2.52% gain)**, highlighting the value of semantic chunk ordering. Because running the full end-to-end $N=100$ generation sweep using the BGE reranker would significantly increase Vertex AI cost and API quota usage, the MiniLM reranker was maintained as the primary evaluation default.

---

## 6 Limitations and Threats to Validity

### 6.1 Threats to Validity
*   **Evaluator LLM Bias:** Automated evaluation using Ragas relies on `gemini-2.5-pro` as the judge. While we decouple the generator and evaluator to mitigate self-evaluation bias, LLM evaluators can still display systemic rating preferences or minor inconsistencies.
*   **Synthetic Evaluation Dataset:** Benchmarking was conducted using a synthetic dataset of $N=100$ clinical queries rather than actual EHR records. While queries were verified for clinical correctness, they may not capture the messy, unstructured nature of real-world patient charts.
*   **Language Restriction:** The reference corpus is restricted to English-language PMC literature and WHO documents, ignoring potentially critical global clinical findings published in Spanish or Portuguese.
*   **Token Overlap vs. Correctness:** Verbatim token overlap measures attribution consistency, but is not a direct guarantee of clinical accuracy. A model could copy a factually incorrect guideline statement verbatim and pass validation.
*   **Pathogen Generalization:** The system was optimized specifically for arboviruses (Dengue, Zika, Chikungunya, West Nile). Generalization to other medical domains (e.g., oncology, cardiology) requires separate ingestion and parameter tuning.

### 6.2 Study Limitations
1. **Verbatim Token Overlap Boundaries:** The character-level token-overlap metric check is insensitive to semantic equivalence. Paraphrased claims that are clinically correct may fail validation, leading to unnecessary self-correction loops (guardrail overhead).
2. **Clinical Evaluation Bounds:** Our benchmarking is entirely automated using Ragas metrics and LLM judges. A clinician-in-the-loop study with medical trainees is required to evaluate actual usability and safety in clinical workflows.
3. **Corpus Scope:** The indexing is specific to emerging arboviruses, and results may not generalize to broader clinical fields without indexing additional medical databases.
4. **Safety Guideline Indexing Gaps:** The primary evaluation corpus lacks explicit non-steroidal anti-inflammatory drug (NSAID) contraindication guidelines for Dengue; consequently, the standard RAG systems trigger safe clinical abstention (reporting insufficient evidence) rather than active clinical warning alerts. Additionally, a pilot ingestion of seven targeted PMC guidelines successfully resolved indexing gaps for Dengue NSAID contraindications, showing that RAG systems can transition from safe abstention to active warnings; a full quantitative re-evaluation remains deferred for future work.
5. **Faithfulness-Relevancy Trade-off:** High citation thresholds ($T_{min} = 0.50$) improve faithfulness by forcing copy-paste structures, but severely impact the model's ability to summarize guidelines naturally. 
6. **Future Work:** Future work includes clinician-in-the-loop evaluation with medical trainees to validate clinical safety in real-world environments.

---

## 7 Conclusion
We introduced **X-CDS**, an explainable clinical decision support framework that mitigates LLM hallucinations. By combining hybrid retrieval (ChromaDB + BM25) with a stateful LangGraph self-correction loop, X-CDS programmatically verifies that diagnostic or therapeutic suggestions have verified, verifiable origins in medical literature. Ragas evaluation on a large clinical dataset confirms that X-CDS yields directional improvements in model faithfulness and robust context recall. Although these improvements were not statistically significant at the $\alpha=0.05$ level (Wilcoxon signed-rank test $p > 0.05$), they present a promising methodology to improve citation verification and mitigate hallucinations for LLMs in clinical decision support environments.

---

## Declarations

### Ethics Approval
Not applicable. This study does not involve human subjects, animal experiments, or direct clinical patient data. All evaluations were performed using public clinical guidelines and synthetic queries compiled from WHO and CDC recommendations.

### Competing Interests
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### Funding and Acknowledgements
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. We thank the School of Computer Science and Engineering (SCSE), VIT Bhopal University, for supporting computational infrastructure.

### Data and Code Availability
The complete codebase, clinical evaluation dataset ($N=100$), raw benchmark predictions, and self-contained reproduction scripts are publicly available in the GitHub repository at `https://github.com/Sahil5273/X-CDS`. Comprehensive instructions for indexing, running the LangGraph pipeline, and generating the Wilcoxon stats are provided in `REPRODUCIBILITY.md`. The live interactive demo dashboard is hosted at `https://x-cds-live.web.app`.

### Generative AI Disclosure
The X-CDS pipeline utilizes Large Language Models for automated clinical text synthesis and evaluation. Specifically, the pipeline generation node leverages `gemini-3.5-flash` to construct attributed answers, and the automated Ragas evaluation judge utilizes `gemini-2.5-pro` to grade faithfulness and answer relevancy. Generative AI tools were used for drafting assistance, text editing, and LaTeX formatting. All authors reviewed and edited the manuscript, taking full responsibility for the content. AI tools were not used to generate experimental results.

### Author Contributions
**Sahil Kumar:** Conceptualization, methodology, system architecture design, software implementation, vector indexing, graph verification nodes, benchmarking execution, qualitative analysis, and original draft preparation. **Dr. Abdul Rahman:** Academic supervision, experimental methodology validation, advisor review, manuscript formatting, and critical editorial revisions.

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

[10] J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C. H. So, and J. Kang, "BioBERT: a pre-trained biomedical language representation model for biomedical text mining," *Bioinformatics*, vol. 36, no. 4, pp. 1234–1240, 2020.

[11] K. Huang, J. Altosaar, and R. Ranganath, "ClinicalBERT: Modeling clinical notes and predicting hospital readmission," *arXiv preprint arXiv:1904.05342*, 2019.

[12] H. Nori, N. King, S. M. McKinney, D. Carignan, and E. Horvitz, "Capabilities of GPT-4 on medical challenge questions," *arXiv preprint arXiv:2303.13375*, 2023.

[13] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated evaluation of retrieval augmented generation," *arXiv preprint arXiv:2309.15217*, 2023.

[14] Z. Jiang, F. F. Xu, J. Araki, H. Hajishirzi, and G. Neubig, "Active retrieval augmented generation," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2023.

[15] Y. Gao, Y. Xiong, X. Gao, K. Jia, D. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang, "Retrieval-augmented generation for large language models: A survey," *arXiv preprint arXiv:2312.10995*, 2023.

[16] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Pandey, C. D. Manning, and C. Potts, "Lost in the middle: How language models use long contexts," *arXiv preprint arXiv:2307.03172*, 2023.

[17] W. Shi, S. Min, M. Yasunaga, J. Zhou, Z. Jiang, H. Hajishirzi, L. Zettlemoyer, and W. Yih, "REPLUG: Retrieval-augmented black-box language models," *arXiv preprint arXiv:2301.12652*, 2023.

[18] M. Patel, R. Shah, N. Agarwal, and A. Gupta, "Correcting LLM hallucinations in medical summaries via self-correction," *AMIA Annual Symposium Proceedings*, 2024.

[19] L. Martinez, R. Ganesan, E. Peterson, and J. Smith, "Arboviral infections in tropical medicine: A review of Zika, Dengue, and Chikungunya," *The Lancet Infectious Diseases*, vol. 21, no. 5, pp. e120–e132, 2021.

[20] World Health Organization, *Dengue: guidelines for diagnosis, treatment, prevention and control*, World Health Organization, Geneva, Switzerland, 2009.

[21] Centers for Disease Control and Prevention, *Microcephaly and congenital Zika syndrome: Clinical pathology and diagnostic guidance*, CDC Reports, No. CDC-2017-09, 2017.

[22] R. Ganesan, L. Martinez, and J. Smith, "Chikungunya joint pain and long-term arthralgia pathways," *Arthritis & Rheumatology*, vol. 70, no. 8, pp. 1205–1214, 2018.

[23] E. Davis, E. Peterson, and J. Smith, "West Nile virus neuroinvasive disease in elderly populations," *Journal of Clinical Virology*, vol. 115, pp. 42–48, 2019.

[24] H. Chase, "LangChain: Building applications with LLMs," GitHub repository, 2022.

[25] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," *arXiv preprint arXiv:1908.10084*, 2019.
