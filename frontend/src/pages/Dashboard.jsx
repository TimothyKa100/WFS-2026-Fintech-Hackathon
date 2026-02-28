// src/pages/Dashboard.jsx
import { useEffect, useMemo, useState } from "react";
import useMarketState from "../hooks/useMarketState";

import Navbar from "../components/layout/Navbar.jsx";
import Panel from "../components/layout/Panel.jsx";

import NetworkGraph from "../components/NetworkGraph.jsx";
import RiskGauge from "../components/RiskGauge.jsx";
import Controls from "../components/Controls.jsx";
import Timeline from "../components/Timeline.jsx";
import CorrelationHeatmap from "../components/CorrelationHeatmap.jsx";

/* ---------- Small inline components (no new file needed) ---------- */

function ScorePill({ value }) {
  if (value == null || !Number.isFinite(Number(value))) return null;
  const n = Number(value);
  const isProb = n >= 0 && n <= 1;
  const text = isProb ? `${Math.round(n * 100)}%` : n.toFixed(2);
  return (
    <span className="text-[11px] px-2 py-0.5 rounded-full border border-border bg-bg/40 text-muted">
      {text}
    </span>
  );
}

function TrendSignals({ up = [], down = [] }) {
  const topUp = (up || []).slice(0, 5);
  const topDown = (down || []).slice(0, 5);

  if (!topUp.length && !topDown.length) return null;

  const Row = ({ item, dir }) => {
    const label = item.label || item.id;
    const score =
      item.prob ?? item.p_up ?? item.p ?? item.score ?? item.signal ?? null;

    return (
      <div className="group relative overflow-hidden rounded-2xl border border-border bg-bg/40 px-3 py-2">
        {/* subtle sheen */}
        <div className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="absolute -inset-x-10 -top-10 h-24 rotate-6 bg-gradient-to-r from-white/0 via-white/5 to-white/0" />
        </div>

        <div className="relative flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${
                dir === "up" ? "bg-emerald-300/90" : "bg-rose-300/90"
              }`}
            />
            <div className="text-sm font-semibold text-text">{label}</div>
          </div>
          <div className="flex items-center gap-2">
            <ScorePill value={score} />
            <span className="text-[11px] text-muted">
              {dir === "up" ? "up" : "down"}
            </span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="mt-4 pt-4 border-t border-border">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold text-text">
          Short-Term Trend (ML)
        </div>
        <div className="text-[11px] text-muted">ranked</div>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-3">
        {topUp.length ? (
          <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
            <div className="text-[11px] text-muted mb-2">Most likely up</div>
            <div className="space-y-2">
              {topUp.map((x) => (
                <Row key={`up-${x.id}`} item={x} dir="up" />
              ))}
            </div>
          </div>
        ) : null}

        {topDown.length ? (
          <div
            className={[
              "rounded-2xl border border-border/60 bg-bg/30 p-3",
              topUp.length ? "" : "",
            ].join(" ")}
          >
            <div className="text-[11px] text-muted mb-2">Most likely down</div>
            <div className="space-y-2">
              {topDown.map((x) => (
                <Row key={`down-${x.id}`} item={x} dir="down" />
              ))}
            </div>
          </div>
        ) : null}
      </div>

      <div className="mt-2 text-[11px] text-muted">
        Scores are model outputs (probability or signal strength).
      </div>
    </div>
  );
}

function VolForecast({ rows = [] }) {
  if (!rows?.length) return null;

  const top = [...rows]
    .map((r) => ({
      ...r,
      _vol:
        r.vol ?? r.vol_forecast ?? r.pred_vol ?? r.sigma ?? r.value ?? null,
    }))
    .filter((r) => r._vol != null && Number.isFinite(Number(r._vol)))
    .sort((a, b) => Number(b._vol) - Number(a._vol))
    .slice(0, 8);

  if (!top.length) return null;

  return (
    <div className="mt-4 pt-4 border-t border-border">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold text-text">
          Volatility Forecast (ML)
        </div>
        <div className="text-[11px] text-muted">top risk</div>
      </div>

      <div className="mt-3 space-y-2">
        {top.map((r) => {
          const label = r.label || r.id;
          const pct = `${(Number(r._vol) * 100).toFixed(1)}%`;
          return (
            <div
              key={r.id}
              className="group relative overflow-hidden flex items-center justify-between rounded-2xl border border-border bg-bg/40 px-3 py-2"
            >
              <div className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="absolute -inset-x-10 -top-10 h-24 rotate-6 bg-gradient-to-r from-white/0 via-white/5 to-white/0" />
              </div>

              <div className="relative text-sm font-semibold text-text">
                {label}
              </div>
              <div className="relative text-[11px] text-muted">
                pred vol{" "}
                <span className="text-text font-semibold">{pct}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-2 text-[11px] text-muted">
        Per-asset predicted volatility (next window).
      </div>
    </div>
  );
}

function TopAnomalies({ topAnomalies = [] }) {
  if (!topAnomalies?.length) return null;

  return (
    <div className="mt-4 pt-4 border-t border-border">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold text-text">Top Anomalies (ML)</div>
        <div className="text-[11px] text-muted">last tick</div>
      </div>

      <div className="mt-3 space-y-2">
        {topAnomalies.slice(0, 5).map((n) => {
          const label = n.label || n.id;
          const score =
            n.anomaly != null ? Math.round(Number(n.anomaly) * 100) : null;
          const ret = n.ret != null ? (Number(n.ret) * 100).toFixed(2) : "—";
          const vol = n.vol != null ? (Number(n.vol) * 100).toFixed(1) : "—";

          return (
            <div
              key={n.id}
              className="group relative overflow-hidden flex items-center justify-between rounded-2xl border border-border bg-bg/40 px-3 py-2"
            >
              <div className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity">
                <div className="absolute -inset-x-10 -top-10 h-24 rotate-6 bg-gradient-to-r from-white/0 via-white/5 to-white/0" />
              </div>

              <div className="relative flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-amber-300/90" />
                <div className="text-sm font-semibold text-text">{label}</div>
              </div>

              <div className="relative text-[11px] text-muted flex items-center gap-3">
                <span>
                  anomaly{" "}
                  <span className="text-text font-semibold">
                    {score != null ? `${score}%` : "—"}
                  </span>
                </span>
                <span className="hidden sm:inline">
                  ret{" "}
                  <span className="text-text font-semibold">
                    {ret !== "—" ? `${ret}%` : "—"}
                  </span>
                </span>
                <span className="hidden sm:inline">
                  vol{" "}
                  <span className="text-text font-semibold">
                    {vol !== "—" ? `${vol}%` : "—"}
                  </span>
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-2 text-[11px] text-muted">
        Node halos reflect anomaly score (0–100%).
      </div>
    </div>
  );
}

/* ---------- Hook: viewport height for full-screen graph ---------- */
function useViewportHeight() {
  const [vh, setVh] = useState(
    typeof window !== "undefined" ? window.innerHeight : 900
  );

  useEffect(() => {
    const onResize = () => setVh(window.innerHeight);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return vh;
}

export default function Dashboard() {
  const { state, nodes, edges, risk, status, regime, forecast, topAnomalies } =
    useMarketState({
      intervalMs: 1200,
      enableMockFallback: true,
    });

  const [controls, setControls] = useState({
    threshold: 0.55,
    window: 30,
    mode: "live",
    playing: true,
    speed: 1,
  });

  const controlsRight = useMemo(() => {
    const t =
      controls?.threshold != null ? Number(controls.threshold).toFixed(2) : "—";
    const w = controls?.window != null ? controls.window : "—";
    return `thr ${t} • win ${w}`;
  }, [controls]);

  const systemRiskRight = useMemo(() => {
    if (status.source === "api") return "live";
    return "demo";
  }, [status.source]);

  const trendUp = forecast?.trend_up ?? forecast?.trendUp ?? forecast?.up ?? [];
  const trendDown =
    forecast?.trend_down ?? forecast?.trendDown ?? forecast?.down ?? [];
  const volForecast =
    forecast?.vol_forecast ?? forecast?.volForecast ?? forecast?.vols ?? [];

  const vh = useViewportHeight();

  // A bit less than 0.85 so the graph doesn't feel cramped under the navbar
  const graphHeight = Math.max(520, Math.round(vh * 0.78));

  return (
    <div className="min-h-screen bg-bg text-text">
      <Navbar status={status} regime={regime} />

      {/* FULL-BLEED CANVAS GRAPH (CLIPPED + vignette) */}
      <div className="relative w-full border-b border-border bg-bg overflow-hidden isolate">
        {/* Vignette/gradient so the graph's own top controls/title look intentional */}
        <div className="pointer-events-none absolute inset-0 z-[1]">
          <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-black/35 to-transparent" />
          <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black/25 to-transparent" />
          <div className="absolute inset-0 [box-shadow:inset_0_0_0_1px_rgba(255,255,255,0.04)]" />
        </div>

        {/* Graph area */}
        <div
          className="relative w-full overflow-hidden z-0"
          style={{ height: graphHeight + 320
           }}
        >
          <NetworkGraph nodes={nodes} edges={edges} height={graphHeight} />
        </div>
      </div>

      {/* BELOW: premium “ops-console” area */}
      <div className="relative">
        {/* ambient background effects */}
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          {/* soft glows */}
          <div className="absolute -top-24 left-1/2 h-64 w-[900px] -translate-x-1/2 rounded-full bg-white/5 blur-3xl" />
          <div className="absolute top-56 -left-32 h-80 w-80 rounded-full bg-white/4 blur-3xl" />
          <div className="absolute top-80 -right-40 h-96 w-96 rounded-full bg-white/3 blur-3xl" />
          {/* faint grid */}
          <div className="absolute inset-0 opacity-[0.05] [background-image:linear-gradient(to_right,rgba(255,255,255,0.6)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.6)_1px,transparent_1px)] [background-size:36px_36px]" />
        </div>

        <div className="relative px-4 sm:px-6 lg:px-8 py-8 space-y-8 max-w-[1600px] mx-auto">
          {/* Header strip (decorative, no functionality) */}
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-2xl border border-border bg-bg/40 grid place-items-center">
                <div className="h-2.5 w-2.5 rounded-full bg-white/70 shadow-[0_0_24px_rgba(255,255,255,0.25)]" />
              </div>
              <div>
                <div className="text-sm font-semibold tracking-wide text-text">
                  Market Contagion Console
                </div>
                <div className="text-[11px] text-muted">
                  Real-time risk, correlations, and ML signals in one view
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-bg/40 text-[11px] text-muted">
                <span
                  className={[
                    "h-2 w-2 rounded-full",
                    status.source === "api"
                      ? "bg-emerald-300/80"
                      : "bg-amber-300/80",
                  ].join(" ")}
                />
                {status.source === "api" ? "Live feed" : "Demo feed"}
              </span>
              <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-bg/40 text-[11px] text-muted">
                <span className="h-2 w-2 rounded-full bg-sky-300/80" />
                rolling window {controls?.window ?? "—"}
              </span>
              <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-bg/40 text-[11px] text-muted">
                <span className="h-2 w-2 rounded-full bg-purple-300/80" />
                threshold {controls?.threshold != null ? Number(controls.threshold).toFixed(2) : "—"}
              </span>
            </div>
          </div>

          {/* 12-col “terminal-style” dashboard grid */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            {/* Left rail: System Risk */}
            <div className="xl:col-span-4">
              <div className="relative">
                {/* rail accent */}
                <div className="pointer-events-none absolute -inset-2 rounded-[28px] bg-gradient-to-b from-white/6 via-transparent to-transparent blur-xl" />
                <div className="relative">
                  <Panel
                    title="System Risk"
                    right={systemRiskRight}
                    className="h-full"
                  >
                    <div className="space-y-3">
                      <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                        <RiskGauge risk={risk} />
                      </div>

                      {/* “module stack” cards */}
                      <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                        <VolForecast rows={volForecast} />
                      </div>

                      <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                        <TrendSignals up={trendUp} down={trendDown} />
                      </div>

                      <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                        <TopAnomalies topAnomalies={topAnomalies} />
                      </div>
                    </div>
                  </Panel>
                </div>
              </div>
            </div>

            {/* Right: Controls + Timeline + Correlation Matrix stacked */}
            <div className="xl:col-span-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="relative">
                  <div className="pointer-events-none absolute -inset-2 rounded-[28px] bg-gradient-to-b from-white/5 via-transparent to-transparent blur-xl" />
                  <div className="relative">
                    <Panel
                      title="Controls"
                      right={controlsRight}
                      className="h-full"
                    >
                      <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                        <Controls initial={controls} onChange={setControls} />
                      </div>

                      <div className="mt-3 text-[11px] text-muted flex items-center justify-between gap-3">
                        <span className="inline-flex items-center gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-white/60" />
                          Tune sensitivity + replay speed
                        </span>
                        <span className="hidden sm:inline text-muted">
                          changes apply instantly
                        </span>
                      </div>
                    </Panel>
                  </div>
                </div>

                <div className="relative">
                  <div className="pointer-events-none absolute -inset-2 rounded-[28px] bg-gradient-to-b from-white/5 via-transparent to-transparent blur-xl" />
                  <div className="relative">
                    <Panel title="Timeline" className="h-full">
                      <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                        <Timeline risk={risk} historyFromApi={state?.history} />
                      </div>

                      <div className="mt-3 text-[11px] text-muted flex items-center justify-between gap-3">
                        <span className="inline-flex items-center gap-2">
                          <span className="h-1.5 w-1.5 rounded-full bg-white/60" />
                          Watch regime shifts over ticks
                        </span>
                        <span className="hidden sm:inline text-muted">
                          zoom for detail
                        </span>
                      </div>
                    </Panel>
                  </div>
                </div>
              </div>

              {/* Correlation Matrix below Controls + Timeline */}
              <div className="mt-6 relative">
                <div className="pointer-events-none absolute -inset-2 rounded-[28px] bg-gradient-to-b from-white/5 via-transparent to-transparent blur-xl" />
                <div className="relative">
                  <Panel title="Correlation Matrix" right="rolling corr">
                    <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                      <CorrelationHeatmap corrMatrix={state?.corr_matrix} />
                    </div>

                    <div className="mt-3 text-[11px] text-muted flex flex-wrap items-center justify-between gap-2">
                      <span className="inline-flex items-center gap-2">
                        <span className="h-1.5 w-1.5 rounded-full bg-white/60" />
                        Dense blocks indicate clustered contagion
                      </span>
                      <span className="text-muted">
                        tip: lower threshold to reveal structure
                      </span>
                    </div>
                  </Panel>
                </div>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="rounded-2xl border border-border bg-bg/30 px-4 py-3 flex flex-wrap items-center justify-between gap-2">
            <div className="text-[11px] text-muted">
              Source:{" "}
              <span className="text-text font-semibold">
                {status.source === "api" ? "Backend API" : "Mock fallback"}
              </span>
            </div>
            <div className="text-[11px] text-muted">
              Tip: use “REPLAY” and lower threshold to see structure changes
              faster.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}