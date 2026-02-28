// src/components/RiskGauge.jsx
function pct(x) {
  const v = Number.isFinite(x) ? x : 0;
  return Math.round(Math.max(0, Math.min(1, v)) * 100);
}

function Bar({ label, value = 0, hint }) {
  const p = pct(value);

  // simple “severity” tone via Tailwind utility classes
  const tone =
    p >= 75
      ? "bg-red-500/70"
      : p >= 50
      ? "bg-amber-400/70"
      : "bg-green-400/70";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="text-xs text-muted">{label}</div>
        <div className="text-xs font-semibold text-text">{p}%</div>
      </div>

      <div className="h-3 rounded-full bg-panel2 border border-border overflow-hidden">
        <div className={["h-full", tone].join(" ")} style={{ width: `${p}%` }} />
      </div>

      {hint ? <div className="text-[11px] text-muted">{hint}</div> : null}
    </div>
  );
}

export default function RiskGauge({ risk }) {
  const pca = risk?.pca ?? 0;

  return (
    <div className="space-y-4">
      <Bar
        label="System Risk (PCA)"
        value={pca}
        hint="Higher = market moving more as one factor (correlation concentration)."
      />

      <div className="pt-2 border-t border-border">
        <div className="text-[11px] text-muted">
          Interpretation: spikes usually mean correlations are rising across the
          universe (contagion).
        </div>
      </div>
    </div>
  );
}