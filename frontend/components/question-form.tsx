"use client";

type QuestionFormProps = {
  question: string;
  onQuestionChange: (question: string) => void;
  onSubmit: () => void;
  loading: boolean;
};

export function QuestionForm({
  question,
  onQuestionChange,
  onSubmit,
  loading,
}: QuestionFormProps) {
  return (
    <form
      className="question-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <textarea
        aria-label="Question for the transcripts"
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        placeholder="Ask about adoption, barriers, budgets, training, or purchasing timelines..."
      />
      <div className="form-footer">
        <span className="muted">Answers are grounded in the indexed interviews.</span>
        <button className="button" type="submit" disabled={loading || !question.trim()}>
          {loading ? "Searching…" : "Ask the transcripts"}
        </button>
      </div>
    </form>
  );
}
