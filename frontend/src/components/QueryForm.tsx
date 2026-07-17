import { useId, type FormEvent } from "react";

type QueryFormProps = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onDemo: () => void;
};

export function QueryForm({
  value,
  loading,
  onChange,
  onSubmit,
  onDemo,
}: QueryFormProps) {
  const fieldId = useId();

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <form className="space-y-3" onSubmit={handleSubmit}>
      <label className="block text-sm font-medium text-[var(--muted)]" htmlFor={fieldId}>
        Symptom or clinical question
      </label>
      <textarea
        id={fieldId}
        className="min-h-28 w-full resize-y rounded-xl border border-[var(--line)] bg-white/70 px-4 py-3 text-[0.98rem] leading-7 text-[var(--ink)] outline-none transition focus:border-[var(--accent)]"
        placeholder="e.g. How do ACE inhibitors help in hypertension?"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      <div className="flex flex-wrap gap-3">
        <button
          type="submit"
          disabled={loading || !value.trim()}
          className="rounded-xl bg-[var(--accent)] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--accent-deep)] disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Retrieving…" : "Ask X-CDS"}
        </button>
        <button
          type="button"
          onClick={onDemo}
          disabled={loading}
          className="rounded-xl border border-[var(--line)] bg-white/60 px-5 py-2.5 text-sm font-semibold text-[var(--ink)] transition hover:border-[var(--accent)] disabled:opacity-50"
        >
          Load demo answer
        </button>
      </div>
    </form>
  );
}
