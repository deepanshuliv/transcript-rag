import type { IndexStatus as IndexStatusData } from "@/lib/api";

type IndexStatusProps = {
  status: IndexStatusData | null;
  loading: boolean;
  onRefresh: () => void;
  onIngest: () => void;
};

export function IndexStatus({ status, loading, onRefresh, onIngest }: IndexStatusProps) {
  return (
    <section className="card">
      <div className="card-header">
        <div>
          <p className="eyebrow">Local knowledge base</p>
          <h2>Index status</h2>
        </div>
        {status && (
          <span className={`status-pill ${status.ready ? "ready" : ""}`}>
            <span className="status-dot" />
            {status.ready ? "Ready" : "Not indexed"}
          </span>
        )}
      </div>
      <div className="card-body">
        {!status ? (
          <p className="muted">Checking the FastAPI backend…</p>
        ) : (
          <>
            <div className="status-details">
              <span><strong>{status.chunk_count}</strong> timestamped chunks</span>
              <span><strong>{status.source_files.length}</strong> source file{status.source_files.length === 1 ? "" : "s"}</span>
              {status.last_indexed_at && <span>Last indexed {new Date(status.last_indexed_at).toLocaleString()}</span>}
              {status.source_files.map((sourceFile) => <span className="source-file" key={sourceFile}>{sourceFile}</span>)}
            </div>
            <div className="form-footer">
              <button className="button secondary" type="button" onClick={onRefresh} disabled={loading}>Refresh</button>
              <button className="button" type="button" onClick={onIngest} disabled={loading}>
                {loading ? "Indexing…" : status.ready ? "Re-index transcripts" : "Index transcripts"}
              </button>
            </div>
          </>
        )}
      </div>
    </section>
  );
}
