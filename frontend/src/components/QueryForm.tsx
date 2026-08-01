import { useId, type FormEvent } from "react";

const PRESETS = [
  {
    label: "Select a clinical preset query...",
    value: "",
  },
  {
    label: "Dengue NSAID Hemorrhagic Risk Warning",
    value: "A patient presents with acute onset of high fever, maculopapular rash, and severe joint pain after travel to India. What NSAID risk must be considered if this is Dengue?",
  },
  {
    label: "Zika Virus Maternal-Fetal Pregnancy Screening",
    value: "A pregnant patient in her first trimester is diagnosed with Zika virus. What fetal complications should be screened for?",
  },
  {
    label: "Dengue Shock Syndrome Critical Warning Signs",
    value: "What hematological and fluid balance changes warn of progression to Dengue Shock Syndrome (DSS)?",
  },
];

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
