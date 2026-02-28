// src/pages/Dashboard.jsx
import { useMemo, useState } from "react";
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
      <div className="flex items-center justify-between rounded-xl border border-border bg-bg/40 px-3 py-2">
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
    );
  };

  return (
    <div className="mt-4 pt-4 border-t border-border">
      <div className="flex items-center justify-between">
        <div className="text-xs font-semibold text-text">Short-Term Trend (ML)</div>
        <div className="text-[11px] text-muted">ranked</div>
      </div>

      <div className="mt-2 grid grid-cols-1 gap-3">
        {topUp.length ? (
          <div>
            <div className="text-[11px] text-muted mb-2">Most likely up</div>
            <div className="space-y-2">
              {topUp.map((x) => (
                <Row key={`up-${x.id}`} item={x} dir="up" />
              ))}
            </div>
          </div>
        ) : null}

        {topDown.length ? (
          <div className={topUp.length ? "pt-2 border-t border-border" : ""}>
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
        <div className="text-xs font-semibold text-text">Volatility Forecast (ML)</div>
        <div className="text-[11px] text-muted">top risk</div>
      </div>

      <div className="mt-2 space-y-2">
        {top.map((r) => {
          const label = r.label || r.id;
          const pct = `${(Number(r._vol) * 100).toFixed(1)}%`;
          return (
            <div
              key={r.id}
              className="flex items-center justify-between rounded-xl border border-border bg-bg/40 px-3 py-2"
            >
              <div className="text-sm font-semibold text-text">{label}</div>
              <div className="text-[11px] text-muted">
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

      <div className="mt-2 space-y-2">
        {topAnomalies.slice(0, 5).map((n) => {
          const label = n.label || n.id;
          const score =
            n.anomaly != null ? Math.round(Number(n.anomaly) * 100) : null;
          const ret = n.ret != null ? (Number(n.ret) * 100).toFixed(2) : "—";
          const vol = n.vol != null ? (Number(n.vol) * 100).toFixed(1) : "—";

          return (
            <div
              key={n.id}
              className="flex items-center justify-between rounded-xl border border-border bg-bg/40 px-3 py-2"
            >
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-amber-300/90" />
                <div className="text-sm font-semibold text-text">{label}</div>
              </div>

              <div className="text-[11px] text-muted flex items-center gap-3">
                <span>
                  anomaly{" "}
                  <span className="text-text font-semibold">
                    {score != null ? `${score}%` : "—"}
                  </span>
                </span>
                <span>
                  ret{" "}
                  <span className="text-text font-semibold">
                    {ret !== "—" ? `${ret}%` : "—"}
                  </span>
                </span>
                <span>
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

  return (
    <div className="min-h-screen bg-bg text-text">
      <Navbar status={status} regime={regime} />

      <div className="mx-auto max-w-[1600px] px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* HERO: big centered graph */}
        <Panel
          title="Contagion Network"
          right={`${nodes.length} assets • ${edges.length} edges`}
          className="overflow-hidden"
        >
          {/* Give it real presence on screen */}
          <div className="w-full flex items-center justify-center">
            <NetworkGraph nodes={nodes} edges={edges} height={720} />
          </div>

          {/* Optional: subtle hint row under hero (keeps same meaning as your legend line in screenshots) */}
          <div className="mt-4 pt-4 border-t border-border text-[11px] text-muted flex flex-wrap gap-x-4 gap-y-2">
            <span>• Positive corr</span>
            <span>• Negative corr (dashed)</span>
            <span>• Anomaly halo</span>
            <span>• Correlation spike (pulse)</span>
            <span>• Edges: top 3 per asset</span>
          </div>
        </Panel>

        {/* BELOW: everything else, aligned + consistent */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* System Risk + ML */}
          <div className="lg:col-span-4">
            <Panel title="System Risk" right={systemRiskRight} className="h-full">
              <RiskGauge risk={risk} />
              <VolForecast rows={volForecast} />
              <TrendSignals up={trendUp} down={trendDown} />
              <TopAnomalies topAnomalies={topAnomalies} />
            </Panel>
          </div>

          {/* Controls */}
          <div className="lg:col-span-4">
            <Panel title="Controls" right={controlsRight} className="h-full">
              <Controls initial={controls} onChange={setControls} />
            </Panel>
          </div>

          {/* Timeline */}
          <div className="lg:col-span-4">
            <Panel title="Timeline" className="h-full">
              <Timeline risk={risk} historyFromApi={state?.history} />
            </Panel>
          </div>

          {/* Correlation heatmap: full width row under those three, still “beneath” hero */}
          <div className="lg:col-span-12">
            <Panel title="Correlation Matrix" right="rolling corr">
              <CorrelationHeatmap corrMatrix={state?.corr_matrix} />
            </Panel>
          </div>
        </div>

        {/* Footer */}
        <div className="text-[11px] text-muted flex flex-wrap items-center justify-between gap-2 pb-6">
          <div>
            Source:{" "}
            <span className="text-text font-semibold">
              {status.source === "api" ? "Backend API" : "Mock fallback"}
            </span>
          </div>
          <div className="text-muted">
            Tip: use “REPLAY” and lower threshold to see structure changes faster.
          </div>
        </div>
      </div>
    </div>
  );
}