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

export type GuideQuestionResult = {
  question_id: string;
  question: string;
  result: VerifiedAnswer;
};

export type ExpertGuideResult = {
  country: string;
  expert: string;
  source_file: string;
  answers: GuideQuestionResult[];
};

export type GuideResponse = {
  countries: string[];
  experts: ExpertGuideResult[];
};

export type IngestResponse = IndexStatus & {
  indexed: boolean;
};

export type StreamStatusEvent = {
  type: "status";
  stage: string;
  label: string;
  detail: string;
};

export type StreamCitationsEvent = {
  type: "citations";
  citations: Citation[];
};

export type StreamDeltaEvent = {
  type: "delta";
  text: string;
};

export type StreamResetEvent = {
  type: "reset";
};

export type StreamTimingEvent = {
  type: "timing";
  stage: string;
  label: string;
  elapsed_ms: number;
  phase?: "started" | "finished" | "failed";
  details: Record<string, unknown>;
};

export type StreamDoneEvent = {
  type: "done";
  answer: VerifiedAnswer;
  timings?: StreamTimingEvent[];
};

export type StreamErrorEvent = {
  type: "error";
  message: string;
};

export type StreamEvent = StreamStatusEvent | StreamCitationsEvent | StreamDeltaEvent | StreamResetEvent | StreamTimingEvent | StreamDoneEvent | StreamErrorEvent;

async function request<T>(path: string, init?: RequestInit, timeoutMs = 15_000): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });

    if (!response.ok) {
      throw new Error("The request could not be completed. Please try again.");
    }

    return (await response.json()) as T;
  } catch (requestError) {
    if (requestError instanceof DOMException && requestError.name === "AbortError") {
      throw new Error(`The archive did not respond within ${Math.round(timeoutMs / 1000)} seconds.`);
    }
    throw requestError;
  } finally {
    clearTimeout(timeout);
  }
}

export function getStatus(): Promise<IndexStatus> {
  return request<IndexStatus>("/api/status", { cache: "no-store" }, 5_000);
}

export function ingestTranscripts(sourceFiles?: string[]): Promise<IngestResponse> {
  const body = sourceFiles ? JSON.stringify({ source_files: sourceFiles }) : undefined;
  return request<IngestResponse>("/api/ingest", {
    method: "POST",
    body,
  }, 240_000);
}

export function askTranscripts(question: string): Promise<VerifiedAnswer> {
  return request<VerifiedAnswer>("/api/ask", {
    method: "POST",
    body: JSON.stringify({ question }),
  }, 120_000);
}

export async function streamTranscripts(
  question: string,
  onEvent: (event: StreamEvent) => void,
  options: { countryScope?: string[]; requireAllCountries?: boolean } = {},
): Promise<VerifiedAnswer> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);

  try {
    const response = await fetch(`${API_BASE_URL}/api/ask/stream`, {
      method: "POST",
      body: JSON.stringify({
        question,
        ...(options.countryScope ? { country_scope: options.countryScope } : {}),
        ...(options.requireAllCountries ? { require_all_countries: true } : {}),
      }),
      signal: controller.signal,
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    });

    if (!response.ok) {
      throw new Error("The request could not be completed. Please try again.");
    }
    if (!response.body) {
      throw new Error("The request could not be completed. Please try again.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finalAnswer: VerifiedAnswer | null = null;

    function consume(lines: string[]) {
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const event = JSON.parse(line.slice(6)) as StreamEvent;
        if (event.type === "error") {
          throw new Error("The request could not be completed. Please try again.");
        }
        onEvent(event);
        if (event.type === "done") finalAnswer = event.answer;
      }
    }

    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      consume(lines);
      if (done) break;
    }
    if (buffer.trim()) consume(buffer.split("\n"));
    if (!finalAnswer) {
      throw new Error("The request could not be completed. Please try again.");
    }
    return finalAnswer;
  } catch (requestError) {
    if (requestError instanceof DOMException && requestError.name === "AbortError") {
      throw new Error("The request took too long. Please try again.");
    }
    throw new Error("The request could not be completed. Please try again.");
  } finally {
    clearTimeout(timeout);
  }
}

export function fetchInterviewGuide(): Promise<GuideResponse> {
  return request<GuideResponse>("/api/guide", { method: "POST" }, 240_000);
}

export function compareAllMarkets(): Promise<VerifiedAnswer> {
  return request<VerifiedAnswer>("/api/compare", { method: "POST" }, 180_000);
}
