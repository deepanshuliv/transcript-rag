const GUIDE_QUESTIONS = [
  "How would you describe current adoption of robotic surgery in your market?",
  "What are the main barriers to adoption?",
  "How important are hospital budgets and ROI in purchasing decisions?",
  "How important are surgeon training and clinical outcomes?",
  "What adoption trend do you expect over the next 3–5 years?",
  "What is the typical hospital decision-making timeline for purchasing a new robotic system?",
];

type GuideQuestionsProps = {
  onSelect: (question: string) => void;
};

export function GuideQuestions({ onSelect }: GuideQuestionsProps) {
  return (
    <section className="card">
      <div className="card-header">
        <div>
          <p className="eyebrow">Interview guide</p>
          <h2>Try a prompt</h2>
        </div>
      </div>
      <div className="card-body">
        <ul className="guide-list">
          {GUIDE_QUESTIONS.map((question) => (
            <li key={question}>
              <button type="button" onClick={() => onSelect(question)}>{question}</button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
