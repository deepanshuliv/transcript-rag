const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type Citation = {
  evidence_id: string;
  quote: string;
  country: string;
  expert: string;
  source_file: string;
  timestamp: string;
};

export type VerifiedClaim = {
  claim: string;
  evidence_ids: string[];
  citations: Citation[];
  disagreement: boolean;
};

export type VerifiedAnswer = {
  valid: boolean;
  answer: string;
  claims: VerifiedClaim[];
  citations: Citation[];
  abstain: boolean;
  abstain_reason?: string | null;
};

export type IndexStatus = {
  ready: boolean;
  chunk_count: number;
  source_files: string[];
  last_indexed_at?: string | null;
};

export type IngestResponse = IndexStatus & {
  indexed: boolean;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based message when the backend did not return JSON.
    }
    throw new Error(message);
  }

  return (await response.json()) as T;
}

export function getStatus(): Promise<IndexStatus> {
  return request<IndexStatus>("/api/status", { cache: "no-store" });
}

export function ingestTranscripts(sourceFiles?: string[]): Promise<IngestResponse> {
  const body = sourceFiles ? JSON.stringify({ source_files: sourceFiles }) : undefined;
  return request<IngestResponse>("/api/ingest", {
    method: "POST",
    body,
  });
}

export function askTranscripts(question: string): Promise<VerifiedAnswer> {
  return request<VerifiedAnswer>("/api/ask", {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}
