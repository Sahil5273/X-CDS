# X-CDS Technical Concept & Terminology Reference Sheet

This document compiles the core technical concepts, clinical justifications, and academic explanations of the X-CDS system. Use these phrases and definitions to write your manuscript or prepare for review discussions.

---

## 1. Clinical Rationale: The Differential Diagnosis Dilemma

* **The Core Problem:** Emerging arboviruses (Zika, Dengue, Chikungunya, West Nile) and tropical fevers (Malaria, Typhoid, Leptospirosis) present with identical initial symptoms: acute fever, rash, and joint/muscle pain.
* **The High Stakes (The NSAID Trap):** 
  > *“Differentiating overlapping arboviruses is a critical, high-stakes clinical task. If a patient presenting with joint pain is misdiagnosed with Chikungunya and prescribed NSAIDs (such as Aspirin or Ibuprofen), but actually has Dengue, the NSAIDs will impair platelet function and trigger severe, potentially fatal internal hemorrhaging (Dengue Hemorrhagic Fever). This clinical risk demonstrates why LLM hallucinations cannot be tolerated in clinical environments and demands deterministic citation safety.”*
* **The Zika Pregnancy Risk:** 
  > *“Accurate separation of Zika from other febrile pathogens is critical to trigger fetal ultrasound protocols and screen for Congenital Zika Syndrome (CZS) and microcephaly in expectant mothers.”*

---

## 2. Retrieval Engineering: Search Noise & Semantic Interference

* **What is Semantic Noise?** As the reference database expands (to 6,293 passages), multiple documents describing different diseases will use similar words (e.g. "fever", "headache", "rash").
* **Inter-Vector Interference:**
  > *“Scaling the reference database to 6,293 passages introduced significant semantic noise in the 384-dimensional vector space due to overlapping symptoms. While standard dense vector retrieval suffered from cross-pathogen interference (retrieving Dengue documents for Chikungunya queries), our dual-channel RRF and Neural Re-ranking successfully isolated target diagnostic guidelines.”*
* **The Solution (RRF + Cross-Encoder):**
  * **Reciprocal Rank Fusion (RRF):** Merges dense vector search (which captures semantic meaning) and BM25 sparse search (which enforces strict keyword matches like *"Chikungunya"* or *"Leptospirosis"*).
  * **Neural Re-ranking (Cross-Encoder):** Uses a deep transformer (`ms-marco-MiniLM-L-6-v2`) to run a joint query-passage attention check, filtering out the "noise" and placing the most relevant passages at the top of the context window.

---

## 3. RAG Architecture: Chunking Strategies

When indexing documents, how you split the text (chunking) determines the quality of your retrieval:

1. **Fixed-Length Token Chunking (Baseline):** Splits text at a fixed token size (e.g. 250 tokens) with overlap.
   * *Limitation:* Cuts off clinical definitions, guidelines, or tables mid-sentence.
2. **Semantic Chunking (Recommended):** Splits text by measuring the semantic distance between consecutive sentence embeddings and splits the text only when a significant topic shift occurs.
   * *Advantage:* Keeps complete clinical concepts unified in a single passage.
   * *Citation Alignment:* 
     > *“Because our citation guardrail calculates character-level token overlap, Semantic Chunking ensures that the retrieved contexts maintain structural and conceptual unity, enabling the generator to write assertions with high verbatim alignment to the source text.”*
3. **Proposition Chunking:** Uses an LLM to parse sentences into independent atomic statements.
   * *Limitation:* Strips surrounding clinical context needed by the model to synthesize cohesive explanations.

---

## 4. Graph Guardrails: Stateful Self-Correction

* **Deterministic vs. Probabilistic Safety:**
  * Standard RAG relies on the LLM to write truthful responses (probabilistic).
  * X-CDS uses **deterministic safety** by calculating character-level token-overlap:
    \[Overlap(S_{claim}, S_{source}) = \frac{|T(S_{claim}) \cap T(S_{source})|}{|T(S_{claim})|}\]
  * If a cited claim sentence has less than 25% verbatim overlap with the source passage, it is rejected.
* **The LangGraph Feedback Loop:**
  > *“Rather than terminating on validation failures, our stateful LangGraph orchestrator compiles programmatic error logs indicating which claims failed verification and routes the state back to the generation node. This iterative correction loop allows the generator to refine its output, significantly improving model faithfulness.”*

---

## 5. Evaluation Methodology: Decoupling and Bias Mitigation

* **Self-Evaluation Bias:**
  * If the same model (e.g. `gemini-3.5-flash`) generates the answers and judges them during evaluation, it will score itself higher and fail to catch stylistic errors.
* **Decoupled Evaluation:**
  > *“To satisfy clinical reporting standards (TRIPOD+AI) and mitigate self-evaluation bias, we decoupled our generation and evaluation layers. The active pipeline generates clinical responses using `gemini-3.5-flash`, whereas the Ragas benchmarking framework evaluates those responses using a separate, Pro-tier model (`gemini-2.5-pro`) as the objective judge.”*

---

## 6. Database Engineering: SQLite Parameter Limits in Vector Stores

* **The Limitation:** ChromaDB utilizes **SQLite** for metadata storage, which enforces a compile-time variable limit (`SQLITE_LIMIT_VARIABLE_NUMBER`) of **32,766 parameters** per single SQL query.
* **Why Upserts Fail at Scale:**
  During a bulk upsert, the parameter count scales with both the number of passages and vector dimensions:
  \[\text{Parameters per chunk} \approx \text{Dimensions (384)} + \text{ID (1)} + \text{Text (1)} + \text{Metadata (4)} \approx 390\]
  Trying to upsert all 6,293 passages in a single database transaction requires **~2,454,270 parameters**, triggering a database error:
  `chromadb.errors.InternalError: ValueError: Batch size of 6293 is greater than max batch size of 5461`
* **The Batching Solution:**
  > *“To populate our vector store, 6,293 clinical passages were embedded into 384-dimensional dense vectors. Due to database variable binding limits in SQLite, the upsert operations were partitioned and executed in 4 sequential batches of 2,000 items.”*

---

## 7. Model Parameters (Weights) vs. SQL Parameters (Database Variables)

* **SQL Parameters (Database Variables):** The variables in a single database query. (e.g., our 2.45 million query bindings). **Do not** refer to this as the "model's parameters" in your paper.
* **Model Parameters (Weights & Biases):** The learned parameters of the actual neural networks.
  * **Embedding Model (`BAAI/bge-small-en-v1.5`):** **33.5 Million parameters**.
  * **Cross-Encoder Re-ranker (`ms-marco-MiniLM-L-6-v2`):** **22 Million parameters**.
  * **Generator Model (`gemini-3.5-flash`):** Estimated in the **billions of parameters**.
