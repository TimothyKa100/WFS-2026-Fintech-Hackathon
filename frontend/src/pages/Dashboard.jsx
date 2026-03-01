// src/pages/Dashboard.jsx
import { useEffect, useMemo, useState } from "react";
import useMarketState from "../hooks/useMarketState";

import Navbar from "../components/layout/Navbar.jsx";
import Panel from "../components/layout/Panel.jsx";

import NetworkGraph from "../components/NetworkGraph.jsx";
import RiskGauge from "../components/RiskGauge.jsx";
import Timeline from "../components/Timeline.jsx";
import CorrelationHeatmap from "../components/CorrelationHeatmap.jsx";

import { corrDictToMatrix } from "../api/transform";

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
          <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
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

/* ---------- Inline Controls UI (removes LIVE/REPLAY + rolling window) ---------- */
function InlineControls({ controls, setControls }) {
  const threshold =
    controls?.threshold != null && Number.isFinite(Number(controls.threshold))
      ? Number(controls.threshold)
      : 0.55;

  const speed =
    controls?.speed != null && Number.isFinite(Number(controls.speed))
      ? Number(controls.speed)
      : 1;

  const playing = !!controls?.playing;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() =>
            setControls((p) => ({ ...p, playing: !p.playing }))
          }
          className={[
            "px-3 py-1.5 rounded-xl border text-sm font-semibold",
            playing
              ? "border-emerald-400/40 bg-emerald-400/15 text-text"
              : "border-amber-400/40 bg-amber-400/15 text-text",
          ].join(" ")}
        >
          {playing ? "Pause" : "Resume"}
        </button>

        <div className="ml-auto flex items-center gap-2">
          <div className="text-xs text-muted">Speed</div>
          <select
            value={speed}
            onChange={(e) =>
              setControls((p) => ({
                ...p,
                speed: Number(e.target.value) || 1,
              }))
            }
            className="px-2 py-1 rounded-xl border border-border bg-bg/40 text-sm"
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs text-muted">Edge threshold (|corr|)</div>
          <div className="text-xs text-text font-semibold">
            {threshold.toFixed(2)}
          </div>
        </div>
        <input
          type="range"
          min={0}
          max={0.99}
          step={0.01}
          value={threshold}
          onChange={(e) =>
            setControls((p) => ({
              ...p,
              threshold: Number(e.target.value),
            }))
          }
          className="w-full"
        />
      </div>

      <div className="text-[11px] text-muted flex items-center justify-between gap-3">
        <span className="inline-flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-white/60" />
          Pause freezes the view (even if backend keeps updating)
        </span>
        <span className="hidden sm:inline text-muted">
          threshold filters graph edges
        </span>
      </div>
    </div>
  );
}

export default function Dashboard() {
  // ✅ Controls: ONLY threshold + pause + speed (no LIVE/REPLAY, no rolling window)
  const [controls, setControls] = useState({
    threshold: 0.55,
    playing: true,
    speed: 1,
  });

  // Polling interval for the hook (nice-to-have). If your hook ignores it, Pause still works via snapshot freeze.
  const baseIntervalMs = 1200;
  const effectiveIntervalMs = useMemo(() => {
    const sp = Number(controls?.speed ?? 1);
    const speed = Number.isFinite(sp) && sp > 0 ? sp : 1;
    return Math.max(200, Math.round(baseIntervalMs / speed));
  }, [controls?.speed]);

  const hook = useMarketState({
    intervalMs: effectiveIntervalMs,
    enableMockFallback: true,

    // correlation day-by-day replay (ONLY correlation; state stays live)
    replayCorrelation: true,
    corrStartDate: "2026-01-06",
    corrEndDate: "2026-02-28",
    corrStepDays: 1,
  });

  // ---- PAUSE THAT ACTUALLY WORKS: freeze what we DISPLAY ----
  const [frozen, setFrozen] = useState(null);

  useEffect(() => {
    // When playing, continuously refresh the frozen snapshot
    if (controls.playing) {
      setFrozen({
        state: hook.state,
        nodes: hook.nodes,
        edges: hook.edges,
        risk: hook.risk,
        status: hook.status,
        regime: hook.regime,
        forecast: hook.forecast,
        topAnomalies: hook.topAnomalies,
      });
    }
    // when paused, do nothing (keeps last frozen snapshot)
  }, [
    controls.playing,
    hook.state,
    hook.nodes,
    hook.edges,
    hook.risk,
    hook.status,
    hook.regime,
    hook.forecast,
    hook.topAnomalies,
  ]);

  const display = controls.playing ? hook : frozen || hook;

  const {
    state,
    nodes,
    edges,
    risk,
    status,
    regime,
    forecast,
    topAnomalies,
  } = display;

  const controlsRight = useMemo(() => {
    const t =
      controls?.threshold != null ? Number(controls.threshold).toFixed(2) : "—";
    return `thr ${t}`;
  }, [controls]);

  const trendUp = forecast?.trend_up ?? forecast?.trendUp ?? forecast?.up ?? [];
  const trendDown =
    forecast?.trend_down ?? forecast?.trendDown ?? forecast?.down ?? [];
  const volForecast =
    forecast?.vol_forecast ?? forecast?.volForecast ?? forecast?.vols ?? [];

  const vh = useViewportHeight();
  const graphHeight = Math.max(520, Math.round(vh * 0.78));

  // --- Heatmap input normalization ---
  const corrMatrixForHeatmap = useMemo(() => {
    if (state?.corr && typeof state.corr === "object") {
      if (Array.isArray(state.corr.assets) && Array.isArray(state.corr.matrix)) {
        return state.corr;
      }
      return corrDictToMatrix(state.corr);
    }

    const raw = state?.corr_matrix ?? state?.corrMatrix ?? null;
    if (
      raw &&
      typeof raw === "object" &&
      Array.isArray(raw.assets) &&
      Array.isArray(raw.matrix)
    ) {
      return raw;
    }
    if (raw && typeof raw === "object" && !Array.isArray(raw)) {
      return corrDictToMatrix(raw);
    }
    return null;
  }, [state]);

  // ✅ Threshold: filter edges client-side
  const filteredEdges = useMemo(() => {
    const thr =
      controls?.threshold != null && Number.isFinite(Number(controls.threshold))
        ? Number(controls.threshold)
        : 0;

    const getW = (e) => {
      const w =
        e?.weight ??
        e?.corr ??
        e?.value ??
        e?.strength ??
        e?.w ??
        e?.r ??
        0;
      const n = Number(w);
      return Number.isFinite(n) ? n : 0;
    };

    if (!Array.isArray(edges) || !edges.length) return [];
    if (thr <= 0) return edges;

    return edges.filter((e) => Math.abs(getW(e)) >= thr);
  }, [edges, controls?.threshold]);

  return (
    <div className="min-h-screen bg-bg text-text">
      <Navbar status={status} regime={regime} />

      {/* FULL-BLEED CANVAS GRAPH (CLIPPED + vignette) */}
      <div className="relative w-full border-b border-border bg-bg overflow-hidden isolate">
        <div className="pointer-events-none absolute inset-0 z-[1]">
          <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-black/35 to-transparent" />
          <div className="absolute inset-x-0 bottom-0 h-24 bg-gradient-to-t from-black/25 to-transparent" />
          <div className="absolute inset-0 [box-shadow:inset_0_0_0_1px_rgba(255,255,255,0.04)]" />
        </div>

        {/* Graph area */}
        <div
          className="relative w-full overflow-hidden z-0"
          style={{ height: graphHeight + 320 }}
        >
          <NetworkGraph nodes={nodes} edges={filteredEdges} height={graphHeight} />
        </div>
      </div>

      {/* BELOW: premium “ops-console” area */}
      <div className="relative">
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute -top-24 left-1/2 h-64 w-[900px] -translate-x-1/2 rounded-full bg-white/5 blur-3xl" />
          <div className="absolute top-56 -left-32 h-80 w-80 rounded-full bg-white/4 blur-3xl" />
          <div className="absolute top-80 -right-40 h-96 w-96 rounded-full bg-white/3 blur-3xl" />
          <div className="absolute inset-0 opacity-[0.05] [background-image:linear-gradient(to_right,rgba(255,255,255,0.6)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.6)_1px,transparent_1px)] [background-size:36px_36px]" />
        </div>

        <div className="relative px-4 sm:px-6 lg:px-8 py-8 space-y-8 max-w-[1600px] mx-auto">
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
                    status?.source === "api"
                      ? "bg-emerald-300/80"
                      : "bg-amber-300/80",
                  ].join(" ")}
                />
                {status?.source === "api" ? "Live feed" : "Demo feed"}
              </span>

              <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-bg/40 text-[11px] text-muted">
                <span
                  className={[
                    "h-2 w-2 rounded-full",
                    controls.playing ? "bg-emerald-300/80" : "bg-amber-300/80",
                  ].join(" ")}
                />
                {controls.playing ? "running" : "paused"}
              </span>

              <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-bg/40 text-[11px] text-muted">
                <span className="h-2 w-2 rounded-full bg-purple-300/80" />
                threshold{" "}
                {controls?.threshold != null
                  ? Number(controls.threshold).toFixed(2)
                  : "—"}
              </span>

              {/* show corr replay date in header pills */}
              {status?.corrDate ? (
                <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-bg/40 text-[11px] text-muted">
                  <span className="h-2 w-2 rounded-full bg-fuchsia-300/80" />
                  corr {status.corrDate}
                </span>
              ) : null}
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
            <div className="xl:col-span-4">
              <div className="relative">
                <div className="pointer-events-none absolute -inset-2 rounded-[28px] bg-gradient-to-b from-white/6 via-transparent to-transparent blur-xl" />
                <div className="relative">
                  <Panel title="System Risk" className="h-full">
                    <div className="space-y-3">
                      <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                        <RiskGauge risk={risk} />
                      </div>

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

            <div className="xl:col-span-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="relative">
                  <div className="pointer-events-none absolute -inset-2 rounded-[28px] bg-gradient-to-b from-white/5 via-transparent to-transparent blur-xl" />
                  <div className="relative">
                    <Panel title="Controls" right={controlsRight} className="h-full">
                      <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                        <InlineControls controls={controls} setControls={setControls} />
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
                  <Panel
                    title="Correlation Matrix"
                    right={status?.corrDate ? `corr ${status.corrDate}` : "rolling corr"}
                  >
                    <div className="rounded-2xl border border-border/60 bg-bg/30 p-3">
                      <CorrelationHeatmap corrMatrix={corrMatrixForHeatmap} />
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

          <div className="rounded-2xl border border-border bg-bg/30 px-4 py-3 flex flex-wrap items-center justify-between gap-2">
            <div className="text-[11px] text-muted">
              Source:{" "}
              <span className="text-text font-semibold">
                {status?.source === "api" ? "Backend API" : "Mock fallback"}
              </span>
            </div>
            <div className="text-[11px] text-muted">
              Tip: correlation replay is active; edges follow{" "}
              <span className="text-text font-semibold">
                {status?.corrDate ?? "—"}
              </span>
              .
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}