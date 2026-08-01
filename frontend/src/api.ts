export type Citation = {
  index: number;
  label: string;
  chunk_id: string;
  pmcid?: string;
  section?: string;
  source_url?: string;
  excerpt?: string;
};

export type ContextChunk = {
  index: number;
  chunk_id: string;
  text: string;
  pmcid?: string;
  section?: string;
  source_url?: string;
  score?: number;
};

export type QueryResponse = {
  query: string;
  answer: string;
  citations: Citation[];
  contexts: ContextChunk[];
  cited_indices: number[];
  validation_passed: boolean;
  validation_issues: Array<Record<string, unknown>>;
  generation_attempts: number;
  error?: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export async function queryXcds(
  query: string,
  crossEncoderModelName?: string,
  citationMinTokenOverlap?: number
): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      cross_encoder_model_name: crossEncoderModelName || undefined,
      citation_min_token_overlap: citationMinTokenOverlap !== undefined ? citationMinTokenOverlap : undefined,
    }),
  });

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // Keep the status-based message when the body is not JSON.
    }
    throw new Error(detail);
  }

  return (await response.json()) as QueryResponse;
}

export const DEMO_RESPONSE: QueryResponse = {
  query: "What hematological and fluid balance changes warn of progression to Dengue Shock Syndrome (DSS)?",
  answer:
    "Beyond fever and rash, the critical hematological and fluid balance changes that must be monitored closely as warning signs for the onset of Dengue Shock Syndrome (DSS) are:\n\n* **Hematological Changes:** An increase in hematocrit (HCT) concurrent with a rapid decrease in platelet count [1], as well as thrombocytopenia [2].\n* **Fluid Balance Changes:** Clinical fluid accumulation [1] and the rapid onset of capillary leakage [2], which can lead to severe plasma leakage and shock (DSS) [1].",
  citations: [
    {
      index: 1,
      label: "[1]",
      chunk_id: "PMC7114207:passage:3",
      pmcid: "PMC7114207",
      section: "Clinical Warning Signs",
      source_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7114207/",
      excerpt: "An increase in hematocrit (HCT) concurrent with a rapid decrease in platelet count is a warning sign of severe dengue and progression to DSS. Clinical fluid accumulation can also occur.",
    },
    {
      index: 2,
      label: "[2]",
      chunk_id: "PMC8439978:passage:12",
      pmcid: "PMC8439978",
      section: "Pathophysiology",
      source_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC8439978/",
      excerpt: "Thrombocytopenia and systemic capillary leakage are hallmarks of severe dengue leading to plasma leakage and shock.",
    },
  ],
  contexts: [
    {
      index: 1,
      chunk_id: "PMC7114207:passage:3",
      text: "An increase in hematocrit (HCT) concurrent with a rapid decrease in platelet count is a warning sign of severe dengue and progression to DSS. Clinical fluid accumulation can also occur.",
      pmcid: "PMC7114207",
      section: "Clinical Warning Signs",
      source_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC7114207/",
    },
    {
      index: 2,
      chunk_id: "PMC8439978:passage:12",
      text: "Thrombocytopenia and systemic capillary leakage are hallmarks of severe dengue leading to plasma leakage and shock.",
      pmcid: "PMC8439978",
      section: "Pathophysiology",
      source_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC8439978/",
    },
  ],
  cited_indices: [1, 2],
  validation_passed: true,
  validation_issues: [],
  generation_attempts: 1,
  error: null,
};
