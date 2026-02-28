// src/components/layout/Navbar.jsx
function Pill({ children, tone = "neutral", dot = true }) {
  const tones = {
    neutral: "bg-panel2 border-border text-muted",
    good: "bg-green-500/10 border-green-500/30 text-green-300",
    bad: "bg-red-500/10 border-red-500/30 text-red-300",
    warn: "bg-amber-500/10 border-amber-500/30 text-amber-300",
    info: "bg-sky-500/10 border-sky-500/30 text-sky-300",
    purple: "bg-purple-500/10 border-purple-500/30 text-purple-300",
  };

  return (
    <span
      className={[
        "inline-flex items-center gap-2 px-3 py-1 rounded-full border text-xs whitespace-nowrap",
        tones[tone] || tones.neutral,
      ].join(" ")}
      title={typeof children === "string" ? children : undefined}
    >
      {dot ? <span className="h-2 w-2 rounded-full bg-current opacity-80" /> : null}
      {children}
    </span>
  );
}

function fmtPct01(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return "";
  return `${Math.round(v * 100)}%`;
}

function regimeTone(label) {
  const l = String(label || "").toUpperCase();
  if (l.includes("BEAR")) return "bad";
  if (l.includes("BULL")) return "good";
  if (l.includes("NEUT") || l.includes("SIDE") || l.includes("TRANS")) return "warn";
  return "purple";
}

export default function Navbar({
  title = "Market Contagion Dashboard",
  regime, // pass from Dashboard
}) {
  const regimeLabel = regime?.label ?? regime?.name;
  const regimeConf = regime?.confidence ?? regime?.conf;

  return (
    <div className="sticky top-0 z-20 backdrop-blur bg-bg/70 border-b border-border">
      <div className="max-w-[1400px] mx-auto px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-accent/15 border border-accent/30 flex items-center justify-center">
            <div className="h-3 w-3 rounded-full bg-accent" />
          </div>
          <div>
            <div className="text-sm font-semibold text-text">{title}</div>
            <div className="text-xs text-muted">
              Correlation network + ML signals
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {regimeLabel ? (
            <Pill tone={regimeTone(regimeLabel)}>
              REGIME: {String(regimeLabel).toUpperCase()}
              {Number.isFinite(Number(regimeConf))
                ? ` • ${fmtPct01(regimeConf)} conf`
                : ""}
            </Pill>
          ) : null}
        </div>
      </div>
    </div>
  );
}