import { useState, useEffect } from "react";

export type TourStep = {
  targetId: string;
  title: string;
  description: string;
};

type GuidedTourProps = {
  steps: TourStep[];
  active: boolean;
  onClose: () => void;
};

export function GuidedTour({ steps, active, onClose }: GuidedTourProps) {
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [coords, setCoords] = useState<{ top: number; left: number; width: number; height: number } | null>(null);

  // Reset step index when opened
  useEffect(() => {
    if (active) {
      setCurrentStepIndex(0);
    }
  }, [active]);

  // Track target element coordinates dynamically
  useEffect(() => {
    if (!active || steps.length === 0) return;

    const updateCoords = () => {
      const step = steps[currentStepIndex];
      const target = document.getElementById(step.targetId);
      if (target) {
        const rect = target.getBoundingClientRect();
        setCoords({
          top: rect.top + window.scrollY,
          left: rect.left + window.scrollX,
          width: rect.width,
          height: rect.height,
        });
        target.scrollIntoView({ behavior: "smooth", block: "center" });
      } else {
        setCoords(null);
      }
    };

    // Calculate immediately and also on resize/scroll
    updateCoords();
    window.addEventListener("resize", updateCoords);
    window.addEventListener("scroll", updateCoords);

    return () => {
      window.removeEventListener("resize", updateCoords);
      window.removeEventListener("scroll", updateCoords);
    };
  }, [active, currentStepIndex, steps]);

  if (!active || steps.length === 0) return null;

  const currentStep = steps[currentStepIndex];

  // Viewport-fixed bottom-center positioning to prevent off-screen overflow and layout shifts
  const popupStyle: React.CSSProperties = {
    position: "fixed",
    bottom: "74px", // Floats comfortably above the sticky disclaimer footer
    left: "50%",
    transform: "translateX(-50%)",
    width: "calc(100% - 32px)",
    maxWidth: "360px",
  };

  const handleNext = () => {
    if (currentStepIndex < steps.length - 1) {
      setCurrentStepIndex((prev) => prev + 1);
    } else {
      onClose();
    }
  };

  const handleBack = () => {
    if (currentStepIndex > 0) {
      setCurrentStepIndex((prev) => prev - 1);
    }
  };

  return (
    <div className="absolute inset-0 pointer-events-none z-[9999]">
      {/* Dark overlay backdrop with a hole cutout highlighted */}
      {coords ? (
        <div
          className="absolute z-[10000] rounded-2xl border border-sky-400/50 pointer-events-none transition-all duration-300 shadow-[0_0_0_9999px_rgba(15,23,42,0.65),0_0_15px_rgba(56,189,248,0.4)]"
          style={{
            top: coords.top - 6,
            left: coords.left - 6,
            width: coords.width + 12,
            height: coords.height + 12,
          }}
        />
      ) : (
        <div className="fixed inset-0 z-[10000] bg-slate-900/65 pointer-events-auto" />
      )}

      {/* Popover content card */}
      <div
        style={popupStyle}
        className="z-[10001] pointer-events-auto rounded-2xl border border-slate-200 bg-white/95 p-5 shadow-2xl backdrop-blur-md transition-all duration-300 ease-out"
      >
        <header className="mb-2 flex items-center justify-between">
          <span className="text-[10px] font-bold uppercase tracking-wider text-[var(--accent-deep)]">
            Step {currentStepIndex + 1} of {steps.length}
          </span>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 text-xs font-semibold"
          >
            Skip
          </button>
        </header>

        <h4 className="text-sm font-bold text-slate-800 mb-1" style={{ fontFamily: "var(--font-display)" }}>
          {currentStep.title}
        </h4>
        <p className="text-xs leading-relaxed text-slate-600 mb-4">
          {currentStep.description}
        </p>

        <footer className="flex items-center justify-between border-t border-slate-100 pt-3">
          <button
            onClick={handleBack}
            disabled={currentStepIndex === 0}
            className="rounded-lg px-2.5 py-1.5 text-xs font-semibold text-slate-500 hover:bg-slate-50 disabled:opacity-40 disabled:hover:bg-transparent"
          >
            &larr; Back
          </button>
          <button
            onClick={handleNext}
            className="rounded-lg bg-[var(--accent)] px-3 py-1.5 text-xs font-bold text-white transition hover:bg-[var(--accent-deep)]"
          >
            {currentStepIndex === steps.length - 1 ? "Finish" : "Next \u2192"}
          </button>
        </footer>
      </div>
    </div>
  );
}
