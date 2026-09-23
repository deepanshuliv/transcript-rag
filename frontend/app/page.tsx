"use client";

import { useEffect, useRef, useState } from "react";

import {
  compareAllMarkets,
  fetchInterviewGuide,
  getStatus,
  ingestTranscripts,
  streamTranscripts,
  type Citation,
  type ExpertGuideResult,
  type GuideResponse,
  type IndexStatus,
  type StreamEvent,
  type VerifiedAnswer,
} from "@/lib/api";

type Turn = {
  id: number;
  question: string;
  answer: VerifiedAnswer | null;
  state: "loading" | "ready" | "error";
  error?: string;
  stage: string;
  stagePhase: "started" | "finished" | "failed";
  stageStartedAt: number | null;
  stageElapsedMs: number | null;
  streamedAnswer: string;
  streamedCitations: Citation[];
  countryScope?: string[];
  requireAllCountries?: boolean;
};

const INITIAL_STAGE = "Reading your question";
const GUIDE_COUNTRIES = ["France", "Germany", "United Kingdom"];

const GUIDE_QUESTIONS = [
  "How would you describe current adoption of robotic surgery in your market?",
  "What are the main barriers to adoption?",
  "How important are hospital budgets and ROI in purchasing decisions?",
  "How important are surgeon training and clinical outcomes?",
  "What adoption trend do you expect over the next 3–5 years?",
  "What is the typical hospital decision-making timeline for purchasing a new robotic system?",
];

function formatElapsed(milliseconds: number) {
  if (milliseconds < 1) return "<1 ms";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(2)} s`;
}

function LiveElapsed({ startedAt }: { startedAt: number }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const update = () => setElapsed(performance.now() - startedAt);
    update();
    const timer = window.setInterval(update, 100);
    return () => window.clearInterval(timer);
  }, [startedAt]);

  return <span className="stage-time" aria-hidden="true">{elapsed < 1000 ? "working" : formatElapsed(elapsed)}</span>;
}

function CitationMarker({
  index,
  citation,
}: {
  index: number;
  citation: Citation;
}) {
  return (
    <span className="citation-wrap">
      <button className="citation-marker" type="button" aria-label={`Citation ${index}`}>
        {index}
      </button>
      <span className="citation-popover" role="tooltip">
        <span className="citation-source">{citation.country} · {citation.expert}</span>
        <span className="citation-quote">“{citation.quote}”</span>
        <span className="citation-file">{citation.source_file} · {citation.timestamp}</span>
      </span>
    </span>
  );
}

function AnswerText({ answer }: { answer: VerifiedAnswer }) {
  return (
    <div className="answer-copy">
      {answer.abstain ? (
        <p>{answer.abstain_reason ?? "The transcripts do not contain enough verified evidence."}</p>
      ) : answer.claims.length > 0 ? answer.claims.map((claim, index) => (
        <p key={`${claim.claim}-${index}`}>
          {claim.claim}
          {claim.citations.map((citation) => (
            <CitationMarker
              key={citation.evidence_id}
              index={answer.citations.findIndex((item) => item.evidence_id === citation.evidence_id) + 1}
              citation={citation}
            />
          ))}
        </p>
      )) : <p>The archive could not produce a verified answer.</p>}
    </div>
  );
}

function ExpertGuide({ expert }: { expert: ExpertGuideResult }) {
  return (
    <section className="expert-guide">
      <div className="expert-guide-heading">
        <div>
          <p className="eyebrow">{expert.country}</p>
          <h2>{expert.expert}</h2>
        </div>
        <span className="source-file">{expert.source_file}</span>
      </div>
      <div className="guide-answer-list">
        {expert.answers.map(({ question_id, question, result }) => (
          <article className="guide-answer" key={question_id}>
            <h3>{question}</h3>
            {result.abstain ? (
              <p className="guide-abstention">{result.abstain_reason}</p>
            ) : <AnswerText answer={result} />}
          </article>
        ))}
      </div>
    </section>
  );
}

export default function HomePage() {
  const [draft, setDraft] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexLoading, setIndexLoading] = useState(false);
  const [indexError, setIndexError] = useState<string | null>(null);
  const [view, setView] = useState<"ask" | "guide">("ask");
  const [selectedCountry, setSelectedCountry] = useState("all");
  const [guideData, setGuideData] = useState<GuideResponse | null>(null);
  const [guideCountry, setGuideCountry] = useState(GUIDE_COUNTRIES[0]);
  const [guideLoading, setGuideLoading] = useState(false);
  const [guideError, setGuideError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<VerifiedAnswer | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const latestTurn = turns.at(-1);
  const isLoading = latestTurn?.state === "loading";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: latestTurn?.streamedAnswer ? "auto" : "smooth",
      block: "end",
    });
  }, [turns.length, latestTurn?.state, latestTurn?.stage, latestTurn?.streamedAnswer.length]);

  useEffect(() => {
    void getStatus()
      .then(setIndexStatus)
      .catch((requestError) => {
        setIndexError(requestError instanceof Error ? requestError.message : "The backend could not be reached.");
      });
  }, []);

  async function indexTranscripts() {
    setIndexLoading(true);
    setIndexError(null);
    try {
      const status = await ingestTranscripts();
      setIndexStatus(status);
      setGuideData(null);
      setComparison(null);
      setGuideError(null);
      setComparisonError(null);
    } catch (requestError) {
      setIndexError(requestError instanceof Error ? requestError.message : "The transcripts could not be indexed.");
    } finally {
      setIndexLoading(false);
    }
  }

  async function openGuide() {
    setView("guide");
    if (guideData || guideLoading) return;
    setGuideLoading(true);
    setGuideError(null);
    try {
      const data = await fetchInterviewGuide();
      setGuideData(data);
      setGuideCountry(data.countries[0] ?? GUIDE_COUNTRIES[0]);
    } catch (requestError) {
      setGuideError(requestError instanceof Error ? requestError.message : "The interview guide could not be generated.");
    } finally {
      setGuideLoading(false);
    }
  }

  async function reloadGuide() {
    setGuideLoading(true);
    setGuideError(null);
    try {
      const data = await fetchInterviewGuide();
      setGuideData(data);
      setGuideCountry(data.countries[0] ?? GUIDE_COUNTRIES[0]);
    } catch (requestError) {
      setGuideError(requestError instanceof Error ? requestError.message : "The interview guide could not be generated.");
    } finally {
      setGuideLoading(false);
    }
  }

  async function runComparison() {
    setComparisonLoading(true);
    setComparisonError(null);
    try {
      setComparison(await compareAllMarkets());
    } catch (requestError) {
      setComparisonError(requestError instanceof Error ? requestError.message : "The market comparison could not be generated.");
    } finally {
      setComparisonLoading(false);
    }
  }

  function submitQuestion(
    questionOverride?: string,
    options: { countryScope?: string[]; requireAllCountries?: boolean } = {},
  ) {
    const question = (questionOverride ?? draft).trim();
    if (!question || isLoading) return;
    const countryScope = options.countryScope
      ?? (selectedCountry === "all" ? undefined : [selectedCountry]);

    const id = Date.now();
    setDraft("");
    setTurns((current) => [
      ...current,
      {
        id,
        question,
        answer: null,
        state: "loading",
        stage: INITIAL_STAGE,
        stagePhase: "started",
        stageStartedAt: performance.now(),
        stageElapsedMs: null,
        streamedAnswer: "",
        streamedCitations: [],
        countryScope,
        requireAllCountries: options.requireAllCountries,
      },
    ]);

    const handleStreamEvent = (event: StreamEvent) => {
      setTurns((current) => current.map((turn) => {
        if (turn.id !== id) return turn;
        if (event.type === "status") {
          return {
            ...turn,
            stage: event.label,
            stagePhase: "started",
            stageStartedAt: performance.now(),
            stageElapsedMs: null,
          };
        }
        if (event.type === "timing") {
          const started = event.phase === "started" || (!event.phase && event.elapsed_ms <= 0);
          return {
            ...turn,
            stage: event.label,
            stagePhase: started ? "started" : event.phase === "failed" ? "failed" : "finished",
            stageStartedAt: started ? performance.now() : null,
            stageElapsedMs: started ? null : event.elapsed_ms,
          };
        }
        if (event.type === "reset") return { ...turn, streamedAnswer: "" };
        if (event.type === "citations") return { ...turn, streamedCitations: event.citations };
        if (event.type === "delta") return { ...turn, streamedAnswer: turn.streamedAnswer + event.text };
        return turn;
      }));
    };

    void streamTranscripts(question, handleStreamEvent, {
      countryScope,
      requireAllCountries: options.requireAllCountries,
    })
      .then((answer) => {
        setTurns((current) => current.map((turn) => turn.id === id
          ? { ...turn, answer, state: "ready", streamedAnswer: answer.answer, streamedCitations: answer.citations }
          : turn));
      })
      .catch(() => {
        const message = "The request could not be completed. Please try again.";
        setTurns((current) => current.map((turn) => turn.id === id ? { ...turn, state: "error", error: message } : turn));
      });
  }

  function handleComposerKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitQuestion();
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <span>Transcript Intelligence</span>
        <nav className="topbar-nav" aria-label="Main navigation">
          <span className={`index-badge ${indexStatus?.ready ? "ready" : ""}`}>
            {indexStatus?.ready
              ? `${indexStatus.source_files.length} transcripts indexed`
              : indexStatus ? "Transcripts not indexed" : indexError ? "Backend unavailable" : "Checking index"}
          </span>
          <button type="button" onClick={() => void indexTranscripts()} disabled={indexLoading}>
            {indexLoading ? "Indexing…" : indexStatus?.ready ? "Re-index" : "Index transcripts"}
          </button>
          <button className={view === "ask" ? "active" : ""} type="button" onClick={() => setView("ask")}>Ask interviews</button>
          <button className={view === "guide" ? "active" : ""} type="button" onClick={() => void openGuide()}>Interview guide</button>
        </nav>
      </header>

      {view === "guide" ? (
        <section className="guide-screen">
          {indexError && <p className="index-error" role="alert">{indexError}</p>}
          <div className="guide-screen-heading">
            <div>
              <p className="eyebrow">France · Germany · United Kingdom</p>
              <h1>Interview guide</h1>
              <p className="guide-intro">Six questions answered separately for each expert, with source quotes and timestamps.</p>
            </div>
            <button className="outline-button" type="button" onClick={() => void reloadGuide()} disabled={guideLoading}>
              {guideLoading ? "Generating answers…" : guideData ? "Regenerate answers" : "Generate answers"}
            </button>
          </div>

          {guideError && <p className="guide-error" role="alert">{guideError}</p>}
          {guideLoading && !guideData && <div className="guide-loading" role="status">Preparing the expert answers…</div>}

          {guideData && (
            <>
              <div className="market-tabs" role="tablist" aria-label="Choose an expert market">
                {guideData.countries.map((country) => (
                  <button
                    key={country}
                    className={guideCountry === country ? "active" : ""}
                    type="button"
                    role="tab"
                    aria-selected={guideCountry === country}
                    onClick={() => setGuideCountry(country)}
                  >
                    {country}
                  </button>
                ))}
              </div>
              {guideData.experts
                .filter((expert) => expert.country === guideCountry)
                .map((expert) => <ExpertGuide expert={expert} key={`${expert.country}-${expert.expert}`} />)}
            </>
          )}

          <section className="comparison-card">
            <div className="comparison-heading">
              <div>
                <p className="eyebrow">All three markets</p>
                <h2>Shared themes and differences</h2>
              </div>
              <button className="outline-button" type="button" onClick={() => void runComparison()} disabled={comparisonLoading}>
                {comparisonLoading ? "Comparing…" : comparison ? "Refresh comparison" : "Compare all three"}
              </button>
            </div>
            <p className="guide-intro">The comparison uses evidence from every interview and preserves market-specific qualifiers.</p>
            {comparisonError && <p className="guide-error" role="alert">{comparisonError}</p>}
            {comparisonLoading && <div className="guide-loading" role="status">Comparing the complete transcript set…</div>}
            {comparison && <AnswerText answer={comparison} />}
          </section>
        </section>
      ) : (
        <>
      {indexError && <p className="index-error chat-index-error" role="alert">{indexError}</p>}

      <div className="chat-scroll">
        <div className={`chat-stage ${turns.length > 0 ? "conversation-active" : "welcome-stage"}`}>
          {turns.length === 0 ? (
            <section className="welcome-content" aria-labelledby="welcome-title">
              <h1 className="welcome-title" id="welcome-title">Good to see you, Yash.</h1>
              <p className="welcome-subtitle">Ask in English. Choose a market to keep evidence scoped to that expert.</p>
              <div className="question-options" aria-label="Suggested questions">
                {GUIDE_QUESTIONS.map((question) => (
                  <button key={question} type="button" onClick={() => submitQuestion(question)} disabled={Boolean(isLoading)}>
                    {question}
                  </button>
                ))}
              </div>
              <button className="guide-shortcut" type="button" onClick={() => void openGuide()}>
                Open the full interview guide and three-market comparison
              </button>
            </section>
          ) : (
            <div className="thread-list">
              {turns.map((turn) => (
                <article className="turn" key={turn.id}>
                  <p className="question">{turn.question}</p>

                  <div className="response">
                    <div className={`stage-line ${turn.state === "loading" ? "stage-active" : ""}`} role="status" aria-live="polite">
                      {turn.state === "loading" && <span className={`stage-indicator stage-indicator-${turn.stagePhase}`} aria-hidden="true" />}
                      <span className="stage-label">{turn.state === "loading" ? turn.stage : turn.state === "error" ? "Request failed" : "Answer"}</span>
                      {turn.state === "loading" && turn.stagePhase === "started" && turn.stageStartedAt !== null && (
                        <LiveElapsed key={`${turn.id}-${turn.stage}-${turn.stageStartedAt}`} startedAt={turn.stageStartedAt} />
                      )}
                      {turn.state === "loading" && turn.stagePhase !== "started" && turn.stageElapsedMs !== null && (
                        <span className="stage-time" aria-hidden="true">{turn.stagePhase === "failed" ? "failed" : formatElapsed(turn.stageElapsedMs)}</span>
                      )}
                    </div>

                    {turn.state === "loading" && (
                      turn.streamedAnswer ? (
                        <p className="streamed-copy">{turn.streamedAnswer}<span className="stream-caret" /></p>
                      ) : (
                        <div className="response-placeholder" aria-hidden="true">
                          <span />
                          <span />
                          <span />
                        </div>
                      )
                    )}

                    {turn.state === "error" && (
                      <div className="request-error" role="alert">
                        <p>{turn.error}</p>
                        <button type="button" onClick={() => { setDraft(turn.question); setSelectedCountry(turn.countryScope?.[0] ?? "all"); setTurns((current) => current.filter((item) => item.id !== turn.id)); window.requestAnimationFrame(() => composerRef.current?.focus()); }}>Try again</button>
                      </div>
                    )}

                    {turn.state === "ready" && turn.answer && <AnswerText answer={turn.answer} />}
                  </div>
                </article>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </div>

      <div className="composer-wrap">
        <form
          className={`composer-shell ${isLoading ? "generating" : ""}`}
          onSubmit={(event) => { event.preventDefault(); submitQuestion(); }}
        >
          <select
            className="market-select"
            aria-label="Evidence market scope"
            value={selectedCountry}
            onChange={(event) => setSelectedCountry(event.target.value)}
            disabled={Boolean(isLoading)}
          >
            <option value="all">All markets</option>
            {GUIDE_COUNTRIES.map((country) => <option key={country} value={country}>{country}</option>)}
          </select>
          <textarea
            ref={composerRef}
            value={draft}
            rows={1}
            aria-label={isLoading ? "The archive is working" : "Ask the interview archive"}
            placeholder={isLoading ? "The archive is working…" : "Ask the interview archive"}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={handleComposerKeyDown}
          />
          <button className="send-button" type="submit" disabled={!draft.trim() || Boolean(isLoading)}>
            {isLoading ? "Working…" : "Send"}
          </button>
        </form>
      </div>
        </>
      )}
    </main>
  );
}
