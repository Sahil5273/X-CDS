import { Fragment } from "react";

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

  // Parse text segments to extract bold (**text**) and citations ([n])
  function parseTextSegment(text: string): React.ReactNode[] {
    const parts = text.split(/(\*\*.*?\*\*|\[\d+\])/g);
    return parts.map((part, offset) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        const innerText = part.slice(2, -2);
        return (
          <strong key={`bold-${offset}`} className="font-bold text-[var(--ink)]">
            {parseTextSegment(innerText)}
          </strong>
        );
      }
      
      const citationMatch = part.match(/^\[(\d+)\]$/);
      if (citationMatch) {
        const index = Number(citationMatch[1]);
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
            [{index}]
          </button>
        );
      }
      return <Fragment key={`text-${offset}`}>{part}</Fragment>;
    });
  }

  // Parse answer line-by-line to render list items, headings, and paragraphs
  const lines = answer.split("\n");
  const renderedLines = lines.map((line, idx) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return <div key={`empty-${idx}`} className="h-2" />;
    }

    if (line.startsWith("###")) {
      return (
        <h3 key={`h3-${idx}`} className="text-lg font-bold mt-4 mb-2 text-[var(--ink)]" style={{ fontFamily: "var(--font-display)" }}>
          {parseTextSegment(line.replace(/^###\s*/, ""))}
        </h3>
      );
    }

    if (trimmed.startsWith("*") || trimmed.startsWith("-")) {
      const cleaned = line.replace(/^\s*[\*\-]\s*/, "");
      return (
        <li key={`li-${idx}`} className="ml-4 list-disc pl-1 text-[1.02rem] leading-8 text-[var(--ink)]">
          {parseTextSegment(cleaned)}
        </li>
      );
    }

    return (
      <p key={`p-${idx}`} className="text-[1.02rem] leading-8 text-[var(--ink)] mb-3">
        {parseTextSegment(line)}
      </p>
    );
  });

  return (
    <div className="space-y-4">
      <div className="prose max-w-none">
        {renderedLines}
      </div>
      <p className="text-sm text-[var(--muted)] border-t border-[var(--line)] pt-3 mt-4">
        Click a citation number to highlight its mapped evidence chunk.
      </p>
    </div>
  );
}
