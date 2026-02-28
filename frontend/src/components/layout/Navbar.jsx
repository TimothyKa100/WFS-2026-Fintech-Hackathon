// src/components/layout/Navbar.jsx
function Pill({ children, tone = "neutral", dot = true }) {
  const tones = {
    // keep same tone semantics, but make it feel "premium"
    neutral:
      "bg-panel2/60 border-border/70 text-muted ring-1 ring-white/5 shadow-[0_8px_24px_-18px_rgba(0,0,0,0.7)]",
    good:
      "bg-green-500/12 border-green-500/35 text-green-200 ring-1 ring-green-400/10 shadow-[0_10px_30px_-18px_rgba(34,197,94,0.35)]",
    bad:
      "bg-red-500/12 border-red-500/35 text-red-200 ring-1 ring-red-400/10 shadow-[0_10px_30px_-18px_rgba(239,68,68,0.35)]",
    warn:
      "bg-amber-500/12 border-amber-500/40 text-amber-200 ring-1 ring-amber-400/10 shadow-[0_10px_30px_-18px_rgba(245,158,11,0.35)]",
    info:
      "bg-sky-500/12 border-sky-500/40 text-sky-200 ring-1 ring-sky-400/10 shadow-[0_10px_30px_-18px_rgba(14,165,233,0.35)]",
    purple:
      "bg-purple-500/12 border-purple-500/40 text-purple-200 ring-1 ring-purple-400/10 shadow-[0_10px_30px_-18px_rgba(168,85,247,0.35)]",
  };

  return (
    <span
      className={[
        "relative inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border text-[11px] whitespace-nowrap",
        "backdrop-blur-xl",
        "transition-transform duration-200 will-change-transform",
        "hover:-translate-y-[1px] active:translate-y-0",
        tones[tone] || tones.neutral,
      ].join(" ")}
      title={typeof children === "string" ? children : undefined}
    >
      {/* sheen */}
      <span className="pointer-events-none absolute inset-0 rounded-full bg-gradient-to-b from-white/10 to-transparent opacity-70" />
      {dot ? (
        <span className="relative h-2 w-2 rounded-full bg-current opacity-95">
          <span className="absolute -inset-1 rounded-full bg-current opacity-20 blur-[2px]" />
        </span>
      ) : null}
      <span className="relative tracking-wide">{children}</span>
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
    <header className="sticky top-0 z-30">
      {/* Decorative aurora / glow layer */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-24 overflow-hidden">
        <div className="absolute -left-24 -top-24 h-64 w-64 rounded-full bg-accent/20 blur-3xl" />
        <div className="absolute left-1/2 -top-28 h-72 w-72 -translate-x-1/2 rounded-full bg-purple-500/15 blur-3xl" />
        <div className="absolute -right-24 -top-24 h-64 w-64 rounded-full bg-sky-500/15 blur-3xl" />
      </div>

      {/* Glass + gradient border shell */}
      <div className="relative">
        {/* gradient hairline border */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-border to-transparent" />

        {/* main glass panel */}
        <div className="backdrop-blur-2xl bg-bg/55 border-b border-border/70">
          <div className="max-w-[1400px] mx-auto px-4 py-3">
            <div className="flex items-center justify-between gap-4">
              {/* Left: Brand block */}
              <div className="flex items-center gap-4 min-w-0">
                {/* Premium mark */}
                <div className="relative h-11 w-11 flex-shrink-0">
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-accent/35 via-accent/10 to-purple-500/10" />
                  <div className="absolute inset-0 rounded-2xl border border-white/10" />
                  <div className="absolute inset-0 rounded-2xl shadow-[0_16px_40px_-26px_rgba(0,0,0,0.9)]" />
                  <div className="relative h-full w-full rounded-2xl flex items-center justify-center">
                    {/* simple "network" glyph made with dots/lines */}
                    <div className="relative h-5 w-5">
                      <span className="absolute left-0 top-2 h-2 w-2 rounded-full bg-accent" />
                      <span className="absolute right-0 top-0 h-2 w-2 rounded-full bg-sky-400/90" />
                      <span className="absolute right-0 bottom-0 h-2 w-2 rounded-full bg-purple-400/90" />
                      <span className="absolute left-1 top-[10px] h-px w-4 rotate-[-18deg] bg-white/20" />
                      <span className="absolute left-1 top-[10px] h-px w-4 rotate-[18deg] bg-white/20" />
                    </div>
                    <div className="absolute -inset-2 rounded-3xl bg-accent/10 blur-xl" />
                  </div>
                </div>

                {/* Title + sub */}
                <div className="min-w-0">
                  <div className="flex items-center gap-2 min-w-0">
                    <div className="text-sm font-semibold text-text truncate">
                      {title}
                    </div>

                    {/* "live" chip */}
                    <span className="inline-flex items-center gap-1.5 rounded-full border border-border/70 bg-panel/40 px-2.5 py-1 text-[10px] text-muted">
                      <span className="relative h-1.5 w-1.5 rounded-full bg-green-400/90">
                        <span className="absolute -inset-1 rounded-full bg-green-400/20 blur-[2px]" />
                      </span>
                      LIVE
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-xs text-muted truncate">
                    <span className="truncate">Correlation network + ML signals</span>
                    <span className="hidden sm:inline opacity-60">•</span>
                    <span className="hidden sm:inline opacity-80">
                      Real-time risk context
                    </span>
                  </div>
                </div>
              </div>

              {/* Right: status / badges */}
              <div className="flex items-center gap-3 flex-shrink-0">
                {/* subtle divider */}
                <div className="hidden md:block h-8 w-px bg-border/70" />

                {regimeLabel ? (
                  <div className="flex items-center gap-2">
                    <div className="hidden lg:flex flex-col items-end leading-tight">
                      <span className="text-[10px] tracking-[0.18em] text-muted/80">
                        CURRENT REGIME
                      </span>
                      <span className="text-[11px] text-muted/90">
                        Model classification
                      </span>
                    </div>

                    <Pill tone={regimeTone(regimeLabel)}>
                      <span className="font-semibold">
                        {String(regimeLabel).toUpperCase()}
                      </span>
                      {Number.isFinite(Number(regimeConf)) ? (
                        <span className="opacity-85">
                          • {fmtPct01(regimeConf)} conf
                        </span>
                      ) : (
                        ""
                      )}
                    </Pill>
                  </div>
                ) : null}
              </div>
            </div>

            {/* Bottom accent sweep */}
            <div className="mt-3 h-[2px] w-full rounded-full bg-gradient-to-r from-transparent via-accent/30 to-transparent" />
          </div>
        </div>
      </div>
    </header>
  );
}