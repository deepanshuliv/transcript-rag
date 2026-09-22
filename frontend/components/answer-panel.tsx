import type { VerifiedAnswer } from "@/lib/api";

type AnswerPanelProps = {
  answer: VerifiedAnswer | null;
};

export function AnswerPanel({ answer }: AnswerPanelProps) {
  return (
    <section className="card" aria-live="polite">
      <div className="card-header">
        <div>
          <p className="eyebrow">Verified response</p>
          <h2>Answer</h2>
        </div>
        {answer && <span className={`status-pill ${answer.valid ? "ready" : ""}`}>{answer.valid ? "Validated" : "Review"}</span>}
      </div>
      <div className="card-body">
        {!answer ? (
          <div className="answer-empty">Ask a question to see a grounded answer and its supporting evidence.</div>
        ) : (
          <>
            <p className="answer">{answer.answer}</p>
            {answer.abstain && (
              <p className="abstention">
                <strong>Abstained:</strong> {answer.abstain_reason ?? "The transcripts do not contain enough verified evidence."}
              </p>
            )}
            {answer.claims.length > 0 && (
              <div>
                <p className="eyebrow">Claim trail</p>
                <ul className="claim-list">
                  {answer.claims.map((claim) => (
                    <li className="claim" key={`${claim.claim}-${claim.evidence_ids.join("-")}`}>
                      <p className="claim-text">{claim.claim}</p>
                      {claim.disagreement && <p className="muted">The cited interviews contain differing views.</p>}
                      <div>
                        {claim.evidence_ids.map((evidenceId) => (
                          <span className="evidence-id" key={evidenceId}>{evidenceId}</span>
                        ))}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </div>
    </section>
  );
}
