import { useId, type FormEvent } from "react";
import { PRESETS } from "../presets";

type QueryFormProps = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onDemo: () => void;
  onRunLive: () => void;
};

export function QueryForm({
  value,
  loading,
  onChange,
  onSubmit,
  onDemo,
  onRunLive,
}: QueryFormProps) {
  const fieldId = useId();

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form className="space-y-4" onSubmit={handleSubmit}>
      <div className="space-y-2">
        <label className="block text-sm font-medium text-[var(--muted)]">
          Clinical Presets
        </label>
        <select
          onChange={(event) => {
            if (event.target.value) {
              onChange(event.target.value);
            }
          }}
          value={PRESETS.find(p => p.value === value) ? value : ""}
          className="w-full rounded-xl border border-[var(--line)] bg-white/70 px-4 py-2.5 text-[0.92rem] text-[var(--ink)] outline-none focus:border-[var(--accent)]"
        >
          {PRESETS.map((p, idx) => (
            <option key={idx} value={p.value}>
              {p.label}
            </option>
          ))}
        </select>
      </div>

      <div className="space-y-2">
        <label className="block text-sm font-medium text-[var(--muted)]" htmlFor={fieldId}>
          Symptom or clinical question
        </label>
        <textarea
          id={fieldId}
          className="min-h-28 w-full resize-y rounded-xl border border-[var(--line)] bg-white/70 px-4 py-3 text-[0.98rem] leading-7 text-[var(--ink)] outline-none transition focus:border-[var(--accent)]"
          placeholder="e.g. What hematological changes warn of severe Dengue Shock Syndrome?"
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          type="submit"
          disabled={loading || !value.trim()}
          className="rounded-xl bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--accent-deep)] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Processing..." : "Ask X-CDS"}
        </button>
        <button
          type="button"
          onClick={onRunLive}
          disabled={loading}
          className="rounded-xl bg-green-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-green-700 disabled:opacity-50"
        >
          Run live demo
        </button>
        <button
          type="button"
          onClick={onDemo}
          disabled={loading}
          className="rounded-xl border border-[var(--line)] bg-white/60 px-5 py-2.5 text-sm font-semibold text-[var(--ink)] transition hover:border-[var(--accent)] disabled:opacity-50"
        >
          Load offline demo
        </button>
      </div>
    </form>
  );
}
