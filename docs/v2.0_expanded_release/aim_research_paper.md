# Explainable Clinical Decision Support (X-CDS): Hybrid Retrieval and Deterministic Citation Verification for Arboviral Clinical Decision Support

**Author:** Sahil Kumar (Registration Number: 23BAI10224)  
**Affiliation:** School of Computer Science and Engineering (SCSE), VIT Bhopal University, India  
**Advisor & Co-Author:** Dr. Abdul Rahman, Associate Professor  
**Corresponding Author:** sahil.kumar2023@vitbhopal.ac.in  

---

## Highlights (Elsevier Mandatory, Max 85 Characters per Bullet)
* X-CDS pairs hybrid retrieval with LangGraph citation guardrails for clinical RAG.
* Ragas faithfulness reaches 93.37% on 100 arbovirus clinical scenarios (T_min=0.10).
* Guardrails validate 90% of answers on the first pass, averaging 1.10 generation loops.
* Mean citations per answer rise from 2.68 (naive RAG) to 3.09 under X-CDS.
* A targeted guideline pilot closed NSAID contraindication gaps in the indexed corpus.

---

## Abstract
Clinical decision support built on large language models (LLMs) can draft useful summaries and treatment suggestions, yet unsupported or incorrect statements remain a major obstacle to deployment. We present **Explainable Clinical Decision Support (X-CDS)**, a retrieval-augmented pipeline that checks each cited clinical claim against retrieved WHO and PMC passages before release. Retrieval combines dense ChromaDB search and sparse BM25 indexing through reciprocal rank fusion (RRF), followed by cross-encoder re-ranking. A LangGraph state machine generates answers with inline citations and routes failed overlap checks back for revision. On $N=100$ arbovirus-focused scenarios evaluated with Ragas, X-CDS ($T_{min}=0.10$) reached 93.37% faithfulness versus 89.78% for naive dense RAG and 91.70% for hybrid RAG without guardrails. Wilcoxon signed-rank tests did not show significance at $\alpha=0.05$ ($p=0.1236$ vs. naive; $p=0.6132$ vs. hybrid), but guardrail telemetry indicates low overhead: 90% first-pass validation, 1.10 mean generation attempts, and a modest rise in clinical abstention when evidence is thin. These results suggest that lightweight, deterministic citation checks can improve grounding in safety-critical infectious-disease settings without heavy retraining.

**Keywords:** Clinical Decision Support Systems; Retrieval-Augmented Generation (RAG); Hallucination Mitigation; Stateful Self-Correction; Emerging Arboviruses; Citation Verification

---

## 1. Introduction
LLMs such as GPT-4 and Gemini can assist clinicians with literature synthesis, differential diagnosis, and draft care plans. They remain statistical language models, however, and can produce fluent but unsupported statements. In clinical use, an uncited drug recommendation or missed contraindication can cause real harm.

We study this problem in emerging arbovirus care, where overlapping presentations make errors especially likely.

### 1.1. Why Arboviruses?
Our evaluation domain covers **Zika, Dengue, Chikungunya, and West Nile virus**. Fever, rash, and arthralgia overlap across these infections, so early differentiation is difficult. The consequences of a wrong label are not symmetric:

* A patient with severe joint pain may receive NSAIDs (e.g., ibuprofen or aspirin) when Chikungunya is suspected.
* If the true diagnosis is Dengue, NSAIDs can worsen platelet dysfunction and precipitate hemorrhagic complications.
* In pregnancy, missed Zika screening can delay detection of congenital Zika syndrome.

Standard retrieval-augmented generation (RAG) reduces—but does not remove—this risk. Failures typically arise when (1) retrieval omits the passage that carries the critical fact, or (2) the generator paraphrases beyond what the retrieved text supports.

**X-CDS** addresses both retrieval quality and post-generation verification. Our contributions are:

1. A **hybrid retrieval** stack (dense embeddings + BM25, RRF merge, cross-encoder re-ranking).
2. A **LangGraph guardrail loop** that enforces minimum token overlap between cited sentences and source chunks.
3. An **empirical study** on 100 arbovirus clinical scenarios, including threshold sweeps, pipeline telemetry, and qualitative case review.

---

## 2. Related Work

### 2.1. Clinical RAG and Medical QA
RAG grounds LLM outputs in external corpora and is now common in biomedical QA [1]. Benchmarks such as BioASQ [2] and MedQA [3] show that retrieval helps, but overlapping disease profiles still produce retrieval and grounding errors in practice [4].

### 2.2. Citation Grounding and Stateful Guardrails
Self-RAG and related methods ask models to judge whether their own outputs are supported by retrieved context [5]. NLI-based attribution scoring offers another post-hoc check [6]. Stateful orchestrators such as LangGraph allow runtime validation and re-generation when a claim fails alignment [7]. X-CDS follows this line but uses an explicit, low-cost token-overlap rule rather than a learned verifier.

### 2.3. Hallucination Mitigation in Clinical LLMs
Domain tuning (e.g., Med-PaLM) and prompt-level guardrails have improved medical QA [8], yet models still omit citations or assert treatments without evidence [9]. We add a deterministic overlap threshold as a deployable safety layer on top of hybrid retrieval.

---

## 3. Materials and Methods

```mermaid
graph TD
    A[Clinical Query] --> B[Hybrid Search: Dense + Sparse]
    B --> C[Reciprocal Rank Fusion (RRF)]
    C --> D[Cross-Encoder Reranker]
    D --> E[Top-K Clinical Chunks]
    E --> F[LangGraph Orchestrator]
    F --> G[Gemini Generation Node]
    G --> H{Citation Overlap Validator}
    H -- Fail: Overlap < 0.10 --> I[State Correction Feedback]
    I --> G
    H -- Pass: Overlap >= 0.10 --> J[Output to Web Dashboard]
```

### 3.1. Corpus and Guideline Sources
We ingested open-access literature through the NIH BioC API. The reference index includes WHO and PAHO arbovirus guidelines (e.g., `PMC7114207`, `PMC8439978`), including classification tables that separate Dengue with warning signs from severe Dengue.

### 3.2. Chunking
Chunks affect both recall and whether cited text can be matched to a source span. We use **semantic chunking**: consecutive sentences are embedded, and a new chunk starts when inter-sentence cosine distance crosses the 95th percentile of within-document distances. Target length is 1,000 characters with 200-character overlap. This keeps related clinical statements together, which helps the overlap validator map claims back to source paragraphs. A formal chunking ablation is left to future work.

### 3.3. Hybrid Retrieval
Two retrievers run in parallel:

1. **Dense search:** `BAAI/bge-small-en-v1.5` embeddings in ChromaDB (cosine similarity).
2. **Sparse search:** BM25 over the same corpus for exact terms (drug names, serotypes, gene markers).

Ranks are fused with reciprocal rank fusion (RRF):

\[RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}\]

where $M=\{\text{dense}, \text{sparse}\}$, $r_m(d)$ is the rank of document $d$ under method $m$, and $k=60$.

### 3.4. Cross-Encoder Re-ranking
The top $N$ RRF candidates are re-scored with `cross-encoder/ms-marco-MiniLM-L-6-v2`:

\[Score(q, d) = \sigma(\mathbf{W}^T \text{Transformer}([CLS]; q; [SEP]; d))\]

The highest-scoring $K$ chunks form the generation context.

### 3.5. LangGraph Generation and Guardrails
The graph has two nodes:

1. **GeminiGenerationNode** — builds a markdown answer with bracketed citations (e.g., `[1]`) via our `RobustChatVertexAI` wrapper.
2. **CitationGuardrailNode** — for each cited sentence $S_{claim}$ and referenced source $S_{source}$:

\[Overlap(S_{claim}, S_{source}) = \frac{|T(S_{claim}) \cap T(S_{source})|}{|T(S_{claim})|}\]

where $T(S)$ is the alphanumeric token set of sentence $S$. If overlap falls below $T_{min}=0.10$, validation fails, errors are logged, and the graph returns to generation.

---

## 4. Experimental Setup

### 4.1. Evaluation Dataset
We built $N=100$ clinical scenarios spanning Dengue (35%), Zika (30%), Chikungunya (20%), and West Nile virus (15%). Questions and reference answers were drafted from CDC and WHO material and checked for clinical plausibility. Each record contains a question, ground-truth answer, and mapped context passages.

The knowledge base held **6,940 chunks** from **73** open-access PMC articles and WHO documents at evaluation time. Evaluation items were prepared separately from corpus indexing to limit train/test leakage.

### 4.2. Baselines and Metrics
We compared three pipelines:

* **Naive RAG** — dense retrieval only; no re-ranking or guardrails.
* **Hybrid RAG** — BM25 + dense + RRF + cross-encoder re-ranking; no guardrails.
* **X-CDS RAG** — full hybrid stack plus LangGraph self-correction ($T_{min}=0.10$).

Scoring used **Ragas** on Google Cloud Vertex AI (`ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings`) for faithfulness, answer relevancy, context precision, and context recall. The generator was `gemini-3.5-flash`; Ragas judging used `gemini-2.5-pro` to reduce same-model bias.

---

## 5. Results and Discussion

### 5.1. Ragas Benchmark Performance
Table 1 summarizes aggregate scores on the full evaluation set.

**Table 1: Comparative Benchmarking of Retrieval and Generation Architectures ($N=100$)**
| Metric | Naive RAG (Dense Only) | Hybrid RAG (RRF + Rerank) | X-CDS RAG ($T_{min}=0.10$) |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | 89.78% | 91.70% | **93.37%** |
| **Context Precision** | 74.09% | 70.91% | 68.94% |
| **Context Recall** | 74.25% | 70.33% | 71.83% |
| **Answer Relevancy** | **61.17%** | 59.81% | 57.81% |

Per-query comparison shows X-CDS improved faithfulness over naive RAG on **23 of 100** cases (60 tied, 17 worse), mean gain **+3.76%**. The Wilcoxon signed-rank test was not significant ($p=0.1236$). Against hybrid RAG, X-CDS improved on **17 of 100** cases (63 tied, 20 worse), mean gain **+1.60%** ($p=0.6132$). Faithfulness gains therefore appear as a directional trend rather than a statistically confirmed effect at $N=100$.

### 5.2. Parametric Threshold Sweep ($T_{min}$)
We swept $T_{min} \in \{0.10, 0.15, 0.25, 0.50\}$ on the same dataset (Table 2).

**Table 2: Impact of Overlap Threshold ($T_{min}$) on Pipeline Metrics ($N=100$)**
| Overlap Threshold ($T_{min}$) | Ragas Faithfulness | Ragas Answer Relevancy |
| :---: | :---: | :---: |
| **0.00** (Baseline RAG) | 89.78% | **61.17%** |
| **0.10** (X-CDS Default) | **93.37%** *(Peak)* | 57.81% |
| **0.15** (X-CDS Mild) | 90.20% | 59.07% |
| **0.25** (X-CDS Strict) | 89.49% | 57.31% |
| **0.50** (X-CDS Extreme) | 92.41% | 57.82% |

At $T_{min}=0.10$, faithfulness peaks while answer relevancy drops modestly relative to the unguarded baseline. Very strict thresholds ($T_{min}=0.25$) increased retry friction and lowered faithfulness to 89.49%. At $T_{min}=0.50$, faithfulness recovered to 92.41% largely through near-verbatim copying from guidelines, at the cost of natural summarization.

### 5.3. Evaluator Separation
Because Ragas itself is LLM-based, we separated generator and judge models. Generation used `gemini-3.5-flash`; evaluation used `gemini-2.5-pro`. This does not remove all evaluator bias but reduces same-family self-grading.

### 5.4. Pipeline and Guardrail Telemetry
Table 3 reports behavior computed from cached run logs.

**Table 3: Pipeline Telemetry and Guardrail Metrics ($N=100$)**
| Metric | Naive RAG (Dense Only) | Hybrid RAG (No Guardrails) | X-CDS RAG ($T_{min}=0.10$) |
| :--- | :---: | :---: | :---: |
| **Mean Generation Attempts** | 1.00 | 1.00 | **1.10** |
| **First-Pass Validation Rate** | N/A | N/A | **90.00%** |
| **Clinical Abstention Rate** | 10.00% | 12.00% | **14.00%** |
| **Mean Citations per Answer** | 2.68 | 2.86 | **3.09** |

Guardrails added limited latency overhead: most answers passed on the first attempt. Abstention rose from 10.00% (naive) and 12.00% (hybrid) to 14.00% under X-CDS, consistent with refusing to answer when retrieved passages lack support. Citation density also increased (2.68 → 3.09 mean citations per answer).

### 5.5. Qualitative Error Analysis
We manually tagged 15 logged cases into five categories: retrieval miss, correct abstention, guardrail success, guardrail overhead, and high-risk scenario.

**Table 4: Tagged Qualitative Error Analysis Cases ($N=15$)**
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

### 5.6. Case Study Vignettes

#### Vignette 1: Severe Dengue Warning Signs
* **Query:** Beyond fever and rash, which hematological and fluid-balance changes signal progression toward dengue shock syndrome (DSS)?
* **X-CDS output:**
    > Beyond fever and rash, the critical hematological and fluid balance changes that must be monitored closely as warning signs for the onset of Dengue Shock Syndrome (DSS) are:
    > *   **Hematological Changes:** An increase in hematocrit (HCT) concurrent with a rapid decrease in platelet count [1], as well as thrombocytopenia [3].
    > *   **Fluid Balance Changes:** Clinical fluid accumulation [1] and the rapid onset of capillary leakage [3], which can lead to severe plasma leakage and shock (DSS) [1].

#### Vignette 2: Abstention When Context Is Incomplete
* **Query:** Which placental cell type acts as a primary Zika entry route to the fetus, and what is its function?
* **X-CDS output:**
    > Based on the provided passages, there is insufficient evidence to identify the specific placental cell type considered a primary target or "Trojan horse" for Zika virus dissemination, or to describe its function.
    >
    > The closest relevant data from the provided sources indicate:
    > *   Zika virus can cross the placenta to infect the developing fetus, and transplacental transmission has been confirmed by detecting viral proteins and RNA in maternal placental tissue samples [1].
    > *   Vertical transmission of the virus from pregnant women to the fetus is responsible for the epidemic of microcephaly [3].
    > *   Zika virus infection has been documented in the brain and placental tissues of congenitally infected newborns and fetal losses [5].

#### Vignette 3: Guardrail Retry (Illustrative Simulation)
*(Reconstructed from self-correction logs to show the feedback loop.)*
* **Query:** A febrile patient with polyarthralgia may have Dengue or Chikungunya. Is ibuprofen appropriate for pain?
* **Attempt 1 (failed):**
    > Ibuprofen can be administered to reduce joint pain and fever in patients with Chikungunya. It also helps manage joint pain in Dengue patients. [1]
* **Guardrail feedback:**
    > Sentence "It also helps manage joint pain in Dengue patients. [1]" has 0.00% verbatim overlap with Source [1]. Source [1] explicitly states: "NSAIDs such as Ibuprofen and Aspirin are strictly contraindicated in suspected Dengue due to the risk of platelet dysfunction and severe hemorrhage."
* **Attempt 2 (passed):**
    > Ibuprofen can be administered for pain relief in patients confirmed to have Chikungunya [1]. However, in patients where Dengue has not been ruled out, NSAIDs such as Ibuprofen are strictly contraindicated due to the risk of platelet dysfunction and hemorrhagic complications [1].

#### Vignette 4: Ungrounded LLM vs. X-CDS
* **Query:** Is ibuprofen safe for a child with suspected Dengue fever?
* **Zero-shot LLM (no retrieval):**
    > Yes, Ibuprofen is generally safe to administer to children with suspected Dengue fever to manage fever and body aches. The recommended pediatric dose is 5–10 mg/kg every 6 to 8 hours. Monitor the child closely for any signs of rash.
* **X-CDS output:**
    > Based on the retrieved clinical guidelines, NSAIDs such as Ibuprofen are strictly contraindicated in cases of suspected Dengue [1]. Administering NSAIDs during Dengue increases the risk of severe hemorrhagic complications and platelet dysfunction [1]. Acetaminophen (Paracetamol) should be used instead for fever and pain management [3].

Without retrieval, the model recommends a contraindicated drug. X-CDS retrieves the relevant guideline language and blocks unsupported claims through the overlap check.

### 5.7. Latency and Cost
Across $N=100$ queries, first-pass answers (90% of cases) averaged **4.08 s** end-to-end; retries averaged **8.45 s**, for an overall mean of **4.52 s**.

Using published Vertex AI rates for `gemini-3.5-flash` ($0.075 per million input tokens; $0.30 per million output tokens) and typical prompt sizes (~3,200 input, ~350 output tokens), a single consult costs about **$0.000345 USD** (~0.029 INR).

### 5.8. Reranker Comparison
On $N=20$ held-out queries, replacing `ms-marco-MiniLM-L-6-v2` with `BAAI/bge-reranker-v2-m3` raised mean context precision from **66.86%** to **69.38%** (+2.52 points). We kept MiniLM for the main $N=100$ runs to limit API cost.

---

## 6. Limitations and Threats to Validity

### 6.1. Threats to Validity
* **LLM-as-judge bias:** Ragas scores depend on `gemini-2.5-pro`; ratings may drift across prompt batches.
* **Synthetic scenarios:** The $N=100$ set is guideline-derived, not extracted from live EHR notes.
* **English-only corpus:** Spanish- and Portuguese-language arbovirus guidance is not indexed.
* **Overlap ≠ clinical truth:** High overlap confirms textual alignment, not independent medical correctness.
* **Domain scope:** Results target arboviruses; other specialties would need new corpora and tuning.

### 6.2. Study Limitations
1. **Paraphrase sensitivity:** Clinically valid paraphrases can fail the overlap rule and trigger extra loops.
2. **No clinician usability study:** Ragas and manual tagging do not replace evaluation with practicing clinicians.
3. **Corpus coverage:** Performance depends on which guidelines are indexed.
4. **NSAID guideline gap:** The main evaluation corpus under-represented explicit Dengue NSAID contraindications, so baseline RAG often abstained rather than warning. A separate pilot added seven PMC articles (+1,014 passages) and qualitatively restored active warnings; we did not re-run the full $N=100$ Ragas battery on the expanded index.
5. **Faithfulness–relevancy trade-off:** Stricter thresholds raise faithfulness at the cost of fluent summarization.
6. **Next steps:** Clinician-in-the-loop testing and expanded multilingual ingestion are planned follow-ons.

---

## 7. Conclusion
X-CDS combines hybrid retrieval with a lightweight LangGraph citation guardrail for arbovirus clinical decision support. On 100 Ragas-evaluated scenarios, it achieved the highest faithfulness among the three tested pipelines (93.37% at $T_{min}=0.10$), with low retry overhead and more frequent abstention when evidence is weak. Statistical tests did not confirm superiority over baselines at $\alpha=0.05$, so the main takeaway is architectural: explicit, deterministic citation checks offer a practical middle ground between unchecked generation and rigid copy-paste from guidelines. Broader validation—with clinicians, larger corpora, and multilingual sources—remains necessary before clinical deployment.

---

## Declaration of Competing Interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## Declaration of Generative AI and AI-Assisted Technologies in the Writing Process
During the preparation of this work, the authors used generative AI tools to assist with literature organization, prose editing, and formatting of the manuscript draft. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article. No generative AI was used to fabricate experimental results, metrics, or qualitative case outputs reported in this paper.

---

## Acknowledgements
This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. We thank the School of Computer Science and Engineering, VIT Bhopal University, for computational support.

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
