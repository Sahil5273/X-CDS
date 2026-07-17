import { useEffect, useRef } from "react";
import type { ContextChunk } from "../api";

type EvidencePanelProps = {
  contexts: ContextChunk[];
  activeCitation: number | null;
};

export function EvidencePanel({ contexts, activeCitation }: EvidencePanelProps) {
  const activeRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (activeCitation == null) {
      return;
    }
    activeRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [activeCitation]);

  if (!contexts.length) {
    return (
      <p className="text-[var(--muted)] text-[0.95rem] leading-relaxed">
        Retrieved passages will appear here with citation mappings.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {contexts.map((chunk) => {
        const isActive = activeCitation === chunk.index;
        return (
          <article
            key={chunk.chunk_id}
            ref={isActive ? activeRef : undefined}
            className={[
              "rounded-xl border px-4 py-4 transition duration-300",
              isActive
                ? "chunk-flash border-[var(--highlight-ring)] bg-[var(--highlight)]"
                : "border-[var(--line)] bg-white/55",
            ].join(" ")}
            data-chunk-index={chunk.index}
          >
            <header className="mb-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span
                className={[
                  "inline-flex min-w-8 items-center justify-center rounded-md px-2 py-0.5 text-sm font-semibold",
                  isActive
                    ? "bg-[var(--accent)] text-white"
                    : "bg-[var(--panel-strong)] text-[var(--accent-deep)]",
                ].join(" ")}
              >
                [{chunk.index}]
              </span>
              <span className="text-sm font-medium text-[var(--ink)]">
                {chunk.section || "Passage"}
              </span>
              {chunk.pmcid ? (
                <span className="text-xs tracking-wide text-[var(--muted)]">
                  {chunk.pmcid}
                </span>
              ) : null}
            </header>
            <p className="text-[0.95rem] leading-7 text-[var(--ink)]">{chunk.text}</p>
            {chunk.source_url ? (
              <a
                className="mt-3 inline-block text-sm text-[var(--accent-deep)] underline-offset-2 hover:underline"
                href={chunk.source_url}
                target="_blank"
                rel="noreferrer"
              >
                Open source
              </a>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}
