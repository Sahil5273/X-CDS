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

export async function queryXcds(query: string): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE}/api/v1/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
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
  query: "How do ACE inhibitors help in hypertension?",
  answer:
    "ACE inhibitors reduce blood pressure in hypertension [1]. They are commonly considered in first-line pharmacologic management when clinically appropriate [2].",
  citations: [
    {
      index: 1,
      label: "[1]",
      chunk_id: "PMC1:passage:0",
      pmcid: "PMC1",
      section: "Abstract",
      source_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
      excerpt: "ACE inhibitors reduce blood pressure in hypertension.",
    },
    {
      index: 2,
      label: "[2]",
      chunk_id: "PMC1:passage:1",
      pmcid: "PMC1",
      section: "Discussion",
      source_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
      excerpt: "ACE inhibitors are often used in first-line hypertension therapy.",
    },
  ],
  contexts: [
    {
      index: 1,
      chunk_id: "PMC1:passage:0",
      text: "ACE inhibitors reduce blood pressure in hypertension.",
      pmcid: "PMC1",
      section: "Abstract",
      source_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
    },
    {
      index: 2,
      chunk_id: "PMC1:passage:1",
      text: "ACE inhibitors are often used in first-line hypertension therapy.",
      pmcid: "PMC1",
      section: "Discussion",
      source_url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/",
    },
  ],
  cited_indices: [1, 2],
  validation_passed: true,
  validation_issues: [],
  generation_attempts: 1,
  error: null,
};
