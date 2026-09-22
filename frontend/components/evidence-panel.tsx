import type { Citation } from "@/lib/api";

type EvidencePanelProps = {
  citations: Citation[];
};

export function EvidencePanel({ citations }: EvidencePanelProps) {
  return (
    <section className="card">
      <div className="card-header">
        <div>
          <p className="eyebrow">Source material</p>
          <h2>Evidence</h2>
        </div>
        {citations.length > 0 && <span className="status-pill ready">{citations.length} citation{citations.length === 1 ? "" : "s"}</span>}
      </div>
      <div className="card-body">
        {citations.length === 0 ? (
          <p className="muted">Verified quotes will appear here after you ask a question.</p>
        ) : (
          <ul className="evidence-list">
            {citations.map((citation) => (
              <li className="evidence-card" key={citation.evidence_id}>
                <blockquote>“{citation.quote}”</blockquote>
                <div className="evidence-meta">
                  <span><strong>{citation.country}</strong> · {citation.expert}</span>
                  <span>Timestamp: {citation.timestamp}</span>
                  <span className="source-file">{citation.source_file}</span>
                  <span className="evidence-id">{citation.evidence_id}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
