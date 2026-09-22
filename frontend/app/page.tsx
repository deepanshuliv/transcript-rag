"use client";

import { useCallback, useEffect, useState } from "react";

import { AnswerPanel } from "@/components/answer-panel";
import { EvidencePanel } from "@/components/evidence-panel";
import { GuideQuestions } from "@/components/guide-questions";
import { IndexStatus } from "@/components/index-status";
import { QuestionForm } from "@/components/question-form";
import {
  askTranscripts,
  getStatus,
  ingestTranscripts,
  type IndexStatus as IndexStatusData,
  type VerifiedAnswer,
} from "@/lib/api";

export default function HomePage() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<VerifiedAnswer | null>(null);
  const [status, setStatus] = useState<IndexStatusData | null>(null);
  const [loading, setLoading] = useState(false);
  const [statusLoading, setStatusLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshStatus = useCallback(async () => {
    setStatusLoading(true);
    try {
      setStatus(await getStatus());
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not reach the FastAPI backend.");
    } finally {
      setStatusLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
  }, [refreshStatus]);

  async function submitQuestion() {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;
    setLoading(true);
    setError(null);
    try {
      setAnswer(await askTranscripts(trimmedQuestion));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The question could not be answered.");
    } finally {
      setLoading(false);
    }
  }

  async function ingest() {
    setStatusLoading(true);
    setError(null);
    try {
      setStatus(await ingestTranscripts());
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The transcripts could not be indexed.");
    } finally {
      setStatusLoading(false);
    }
  }

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">European robotic surgery market</p>
          <h1>Ask the interviews, keep the evidence.</h1>
          <p className="hero-copy">
            Explore the supplied expert transcripts with answers that retain the country, speaker, timestamp, source file, and exact supporting quote.
          </p>
        </div>
        <div className="hero-mark" aria-hidden="true">↗</div>
      </header>

      <div className="layout-grid">
        <div className="main-column">
          <section className="card">
            <div className="card-header">
              <div>
                <p className="eyebrow">Transcript Q&amp;A</p>
                <h2>Ask the transcripts</h2>
                <p className="card-intro">Start with a broad market question or select one of the guide prompts.</p>
              </div>
            </div>
            <div className="card-body">
              <QuestionForm
                question={question}
                onQuestionChange={setQuestion}
                onSubmit={submitQuestion}
                loading={loading}
              />
            </div>
          </section>

          {error && <p className="error" role="alert">{error}</p>}
          <AnswerPanel answer={answer} />
          <EvidencePanel citations={answer?.citations ?? []} />
        </div>

        <aside className="side-column">
          <IndexStatus
            status={status}
            loading={statusLoading}
            onRefresh={() => void refreshStatus()}
            onIngest={() => void ingest()}
          />
          <GuideQuestions onSelect={setQuestion} />
        </aside>
      </div>
    </main>
  );
}
