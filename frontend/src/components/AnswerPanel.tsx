import { Fragment, type ReactNode } from "react";

const CITATION_PATTERN = /(\[\d+\])/g;

type AnswerPanelProps = {
  answer: string;
  activeCitation: number | null;
  onCitationClick: (index: number) => void;
};

export function AnswerPanel({
  answer,
  activeCitation,
  onCitationClick,
}: AnswerPanelProps) {
  if (!answer.trim()) {
    return (
      <p className="text-[var(--muted)] text-[0.95rem] leading-relaxed">
        Submit a symptom or clinical question to generate an explainable answer.
      </p>
    );
  }

  const parts = answer.split(CITATION_PATTERN);
  const nodes: ReactNode[] = parts.map((part, offset) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) {
      return <Fragment key={`text-${offset}`}>{part}</Fragment>;
    }

    const index = Number(match[1]);
    const isActive = activeCitation === index;
    return (
      <button
        key={`cite-${offset}-${index}`}
        type="button"
        className={[
          "mx-0.5 inline-flex min-w-7 items-center justify-center rounded-md px-1.5 py-0.5 align-baseline text-[0.8rem] font-semibold transition duration-200",
          isActive
            ? "bg-[var(--accent)] text-white"
            : "bg-[var(--highlight)] text-[var(--accent-deep)] hover:bg-[var(--accent)] hover:text-white",
        ].join(" ")}
        onClick={() => onCitationClick(index)}
        aria-pressed={isActive}
        aria-label={`Highlight source chunk ${index}`}
      >
        {part}
      </button>
    );
  });

  return (
    <div className="space-y-3">
      <p className="text-[1.02rem] leading-8 text-[var(--ink)]">{nodes}</p>
      <p className="text-sm text-[var(--muted)]">
        Click a citation number to highlight its mapped evidence chunk.
      </p>
    </div>
  );
}
