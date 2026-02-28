// src/hooks/useMarketState.js
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchState } from "../api/client";

/* -------------------- Helpers -------------------- */
const clamp01 = (x) => Math.max(0, Math.min(1, x));

function computeRegime({ pca = 0, stress = 0 } = {}) {
  // demo-safe fallback (replace when backend provides regime)
  if (stress >= 0.7 || pca >= 0.8) {
    return {
      label: "BEARISH",
      confidence: clamp01(0.65 + 0.35 * Math.max(stress, pca)),
    };
  }
  if (stress <= 0.35 && pca <= 0.45) {
    return {
      label: "BULLISH",
      confidence: clamp01(0.55 + 0.45 * (1 - Math.max(stress, pca))),
    };
  }
  return {
    label: "NEUTRAL",
    confidence: clamp01(0.5 + 0.25 * Math.abs(stress - pca)),
  };
}

function computeAnomalyScore(node) {
  // lightweight heuristic: abnormality ~ (|return| * (1 + vol))
  // returns 0..1
  const r = Math.abs(node?.ret ?? 0);
  const v = node?.vol ?? 0;
  return clamp01(r * 45 * (1 + v)); // tuned for your mock magnitudes
}

function buildEdgeDelta(edges = [], prevEdgesMap) {
  // Adds fields:
  // - absWeight
  // - deltaAbsWeight (vs previous tick)
  // - spiked (used for glow pulse)
  const out = edges.map((e) => {
    const s = typeof e.source === "object" ? e.source.id : e.source;
    const t = typeof e.target === "object" ? e.target.id : e.target;
    const key = `${s}__${t}`;
    const w = Number(e.weight ?? 0);
    const absW = Math.abs(w);

    const prev = prevEdgesMap.get(key);
    const prevAbs = prev ? Math.abs(prev) : 0;
    const deltaAbs = absW - prevAbs;

    // spike = crossed threshold or jumped quickly
    const crossedUp = prevAbs < 0.75 && absW >= 0.75;
    const jumped = deltaAbs >= 0.18;

    return {
      ...e,
      source: s,
      target: t,
      absWeight: absW,
      deltaAbsWeight: deltaAbs,
      spiked: crossedUp || jumped,
      ts: e.ts ?? undefined,
    };
  });

  prevEdgesMap.clear();
  for (const e of out) {
    prevEdgesMap.set(`${e.source}__${e.target}`, Number(e.weight ?? 0));
  }

  return out;
}

/* -------------------- Correlation matrix -> edges -------------------- */
/**
 * Accepts correlation matrix formats and returns dense edges:
 * - corrMatrix: number[][] aligned with nodes list order
 * - corr_matrix: number[][] aligned with nodes list order
 * - corr: { [id]: { [id]: number } } (object-of-objects)
 */
function corrToEdges({ nodes = [], corrMatrix, corr_matrix, corr } = {}) {
  const ids = (nodes || []).map((n) => n.id ?? n.label).filter(Boolean);

  // matrix form
  const mat = corrMatrix ?? corr_matrix ?? null;
  if (Array.isArray(mat) && mat.length && Array.isArray(mat[0])) {
    const edges = [];
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const w = Number(mat?.[i]?.[j]);
        if (!Number.isFinite(w)) continue;
        edges.push({ source: ids[i], target: ids[j], weight: w });
      }
    }
    return edges;
  }

  // object-of-objects form
  if (corr && typeof corr === "object") {
    const edges = [];
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        const a = ids[i];
        const b = ids[j];
        const w = Number(corr?.[a]?.[b] ?? corr?.[b]?.[a]);
        if (!Number.isFinite(w)) continue;
        edges.push({ source: a, target: b, weight: w });
      }
    }
    return edges;
  }

  return null;
}

/* -------------------- Forecast normalization -------------------- */
function normalizeForecast(rawForecast, nodes, { pca, stress }, nowTs) {
  const f = rawForecast && typeof rawForecast === "object" ? rawForecast : {};

  const xgbStressProb =
    f.xgbStressProb ?? f.stress_prob ?? f.p_stress ?? f.pStress ?? null;

  const volRaw =
    f.vol_forecast ?? f.volForecast ?? f.vols ?? f.volatility ?? f.vol ?? null;

  let vol_forecast = [];
  if (Array.isArray(volRaw)) {
    vol_forecast = volRaw
      .map((r) => ({
        id: r.id ?? r.symbol ?? r.ticker,
        label: r.label ?? r.id ?? r.symbol ?? r.ticker,
        vol: r.vol ?? r.value ?? r.pred_vol ?? r.sigma,
      }))
      .filter((r) => r.id);
  } else if (volRaw && typeof volRaw === "object") {
    vol_forecast = Object.entries(volRaw).map(([k, v]) => ({
      id: k,
      label: k,
      vol: v,
    }));
  }

  const upRaw = f.trend_up ?? f.trendUp ?? f.up ?? f.long ?? f.longs ?? null;
  const downRaw =
    f.trend_down ?? f.trendDown ?? f.down ?? f.short ?? f.shorts ?? null;

  const toRankList = (raw) => {
    if (!raw) return [];
    if (Array.isArray(raw)) {
      if (raw.length && typeof raw[0] === "string") {
        return raw.slice(0, 10).map((id, idx) => ({
          id,
          label: id,
          prob: clamp01(0.65 - idx * 0.05),
        }));
      }
      return raw
        .map((r) => ({
          id: r.id ?? r.symbol ?? r.ticker,
          label: r.label ?? r.id ?? r.symbol ?? r.ticker,
          prob:
            r.prob ?? r.p_up ?? r.p_down ?? r.p ?? r.score ?? r.signal ?? null,
        }))
        .filter((r) => r.id);
    }
    if (typeof raw === "object") {
      return Object.entries(raw)
        .map(([id, prob]) => ({ id, label: id, prob }))
        .sort((a, b) => Number(b.prob ?? 0) - Number(a.prob ?? 0))
        .slice(0, 10);
    }
    return [];
  };

  let trend_up = toRankList(upRaw);
  let trend_down = toRankList(downRaw);

  if (!trend_up.length && nodes?.length) {
    const sorted = [...nodes].sort(
      (a, b) => Number(b.ret ?? 0) - Number(a.ret ?? 0)
    );
    trend_up = sorted.slice(0, 5).map((n, i) => ({
      id: n.id,
      label: n.label ?? n.id,
      prob: clamp01(
        0.55 +
          0.35 * clamp01((Number(n.ret ?? 0) + 0.02) / 0.04) -
          i * 0.03
      ),
    }));
  }

  if (!trend_down.length && nodes?.length) {
    const sorted = [...nodes].sort(
      (a, b) => Number(a.ret ?? 0) - Number(b.ret ?? 0)
    );
    trend_down = sorted.slice(0, 5).map((n, i) => ({
      id: n.id,
      label: n.label ?? n.id,
      prob: clamp01(
        0.55 +
          0.35 * clamp01((Math.abs(Number(n.ret ?? 0)) + 0.0) / 0.03) -
          i * 0.03
      ),
    }));
  }

  if (!vol_forecast.length && nodes?.length) {
    vol_forecast = nodes.map((n) => ({
      id: n.id,
      label: n.label ?? n.id,
      vol: n.vol ?? 0,
    }));
  }

  vol_forecast = vol_forecast
    .map((r) => ({ ...r, vol: Number(r.vol) }))
    .filter((r) => Number.isFinite(r.vol))
    .slice(0, 50);

  trend_up = trend_up
    .map((r) => ({ ...r, prob: r.prob == null ? null : Number(r.prob) }))
    .filter((r) => r.id)
    .slice(0, 10);

  trend_down = trend_down
    .map((r) => ({ ...r, prob: r.prob == null ? null : Number(r.prob) }))
    .filter((r) => r.id)
    .slice(0, 10);

  const fallbackStress = clamp01(
    0.15 + 0.75 * (0.6 * (stress ?? 0) + 0.4 * (pca ?? 0))
  );

  return {
    xgbStressProb:
      xgbStressProb == null || !Number.isFinite(Number(xgbStressProb))
        ? fallbackStress
        : clamp01(Number(xgbStressProb)),
    vol_forecast,
    trend_up,
    trend_down,
    ts: nowTs,
  };
}

/* -------------------- 25-asset universe (mock + API cap) -------------------- */
const DEFAULT_UNIVERSE_25 = [
  "BTC",
  "ETH",
  "SOL",
  "XRP",
  "BNB",
  "ADA",
  "AVAX",
  "DOGE",
  "DOT",
  "LINK",
  "MATIC",
  "LTC",
  "BCH",
  "TRX",
  "ATOM",
  "UNI",
  "XLM",
  "FIL",
  "APT",
  "SUI",
  "OP",
  "ARB",
  "INJ",
  "NEAR",
  "ETC",
];

/* -------------------- Mock fallback (demo-safe) -------------------- */
function makeMockState(now = Date.now(), universe = DEFAULT_UNIVERSE_25) {
  const assets = universe.slice(0, 25);

  const t = (now / 1000);
  const nodes = assets.map((a, i) => {
    const vol = 0.18 + 0.18 * Math.abs(Math.sin(t / 23 + i * 0.23));
    const ret = 0.012 * Math.sin(t / 9 + i * 0.31);

    // group for coloring (optional)
    const majorsSet = new Set(["BTC", "ETH", "SOL"]);
    const group = majorsSet.has(a) ? "majors" : "alts";

    return {
      id: a,
      label: a,
      group,
      vol,
      price: 100 + 35 * Math.sin(t / 17 + i * 0.17),
      ret,
      anomaly: 0, // filled below
    };
  });

  // Dense correlations (ALL PAIRS) so NetworkGraph can pick top-3 per node
  const edges = [];
  for (let i = 0; i < assets.length; i++) {
    for (let j = i + 1; j < assets.length; j++) {
      // Smooth, bounded in [-0.95, 0.95]
      const raw =
        Math.sin(t / 11 + i * 0.21 + j * 0.13) * 0.75 +
        Math.sin(t / 37 + i * 0.07 - j * 0.05) * 0.25;
      const weight = Math.max(-0.95, Math.min(0.95, raw));
      edges.push({ source: assets[i], target: assets[j], weight });
    }
  }

  const pca = clamp01(0.25 + 0.5 * Math.abs(Math.sin(t / 19)));
  const stress = clamp01(0.2 + 0.6 * Math.abs(Math.sin(t / 13)));

  for (const n of nodes) n.anomaly = computeAnomalyScore(n);

  const regime = computeRegime({ pca, stress });

  const forecast = normalizeForecast({}, nodes, { pca, stress }, now);

  return {
    ts: now,
    mode: "mock",
    nodes,
    edges,
    risk: { pca, stress },
    regime,
    forecast,
  };
}

/* -------------------- Hook -------------------- */
export default function useMarketState({
  intervalMs = 1200,
  enableMockFallback = true,
  universe = DEFAULT_UNIVERSE_25, // allow override
  maxAssets = 25, // cap API nodes for your "25 assets" requirement
} = {}) {
  const [state, setState] = useState(() =>
    makeMockState(Date.now(), universe)
  );
  const [status, setStatus] = useState({
    ok: true,
    source: "mock", // "api" | "mock"
    lastUpdated: Date.now(),
    error: null,
  });

  const timerRef = useRef(null);
  const backoffRef = useRef(0);

  // keep last edge weights so we can detect "spikes" for glow pulses
  const prevEdgesMapRef = useRef(new Map());

  useEffect(() => {
    let cancelled = false;

    async function tick() {
      try {
        const data = await fetchState();
        if (cancelled) return;

        let nextState = null;

        if (data && typeof data === "object") {
          nextState = data;
        } else if (enableMockFallback) {
          nextState = makeMockState(Date.now(), universe);
        }

        if (nextState) {
          // ---- Nodes (cap to 25 + ensure ids) ----
          let nodes = Array.isArray(nextState.nodes) ? nextState.nodes : [];

          // If backend sends more than 25 nodes, keep the first 25
          // (or apply your own ordering upstream)
          nodes = nodes.slice(0, maxAssets).map((n) => ({
            ...n,
            id: n.id ?? n.symbol ?? n.ticker ?? n.label,
            label: n.label ?? n.id ?? n.symbol ?? n.ticker,
          }));

          // If backend sends none, use mock nodes for UI stability
          if (!nodes.length && enableMockFallback) {
            nodes = makeMockState(Date.now(), universe).nodes;
          }

          const risk = nextState.risk || {};
          const pca = Number(risk.pca ?? 0);
          const stress = Number(risk.stress ?? 0);

          // Ensure anomaly exists even if backend doesn't provide it
          const enrichedNodes = nodes.map((n) => ({
            ...n,
            anomaly: n.anomaly ?? computeAnomalyScore(n),
          }));

          // Ensure regime exists even if backend doesn't provide it
          const regime =
            nextState.regime && typeof nextState.regime === "object"
              ? nextState.regime
              : computeRegime({ pca, stress });

          // ---- Edges ----
          // Prefer backend edges if provided; otherwise, try to derive from correlation matrix.
          let edges = Array.isArray(nextState.edges) ? nextState.edges : [];

          if (!edges.length) {
            const derivedEdges = corrToEdges({
              nodes: enrichedNodes,
              corrMatrix: nextState.corrMatrix,
              corr_matrix: nextState.corr_matrix,
              corr: nextState.corr,
            });
            if (derivedEdges?.length) edges = derivedEdges;
          }

          // If still no edges and mock is allowed, fall back to dense mock edges
          if (!edges.length && enableMockFallback) {
            edges = makeMockState(Date.now(), universe).edges;
          }

          // Enrich with spike fields
          const enrichedEdges = buildEdgeDelta(edges, prevEdgesMapRef.current);

          // Forecast: normalize into full ML-output shape
          const forecast = normalizeForecast(
            nextState.forecast,
            enrichedNodes,
            { pca, stress },
            nextState.ts ?? Date.now()
          );

          setState({
            ...nextState,
            nodes: enrichedNodes,
            edges: enrichedEdges,
            regime,
            forecast,
            ts: nextState.ts ?? Date.now(),
          });

          setStatus({
            ok: !!(data && typeof data === "object"),
            source: data && typeof data === "object" ? "api" : "mock",
            lastUpdated: Date.now(),
            error:
              data && typeof data === "object"
                ? null
                : enableMockFallback
                ? "Empty /state response"
                : "Empty /state response",
          });
        }

        backoffRef.current = 0;
      } catch (e) {
        if (cancelled) return;

        const msg = e?.message || String(e);
        if (enableMockFallback) {
          const mock = makeMockState(Date.now(), universe);
          const edges = buildEdgeDelta(mock.edges, prevEdgesMapRef.current);
          setState({ ...mock, edges });
        }

        setStatus({
          ok: false,
          source: enableMockFallback ? "mock" : "api",
          lastUpdated: Date.now(),
          error: msg,
        });

        backoffRef.current = Math.min(backoffRef.current + 400, 3000);
      } finally {
        if (cancelled) return;
        const next = intervalMs + backoffRef.current;
        timerRef.current = setTimeout(tick, next);
      }
    }

    tick();

    return () => {
      cancelled = true;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [intervalMs, enableMockFallback, universe, maxAssets]);

  const derived = useMemo(() => {
    const nodes = state?.nodes || [];
    const edges = state?.edges || [];
    const risk = state?.risk || {};
    const ts = state?.ts || Date.now();
    const regime = state?.regime || computeRegime(risk);
    const forecast = state?.forecast || {};

    const topAnomalies = [...nodes]
      .filter((n) => Number.isFinite(n.anomaly))
      .sort((a, b) => (b.anomaly ?? 0) - (a.anomaly ?? 0))
      .slice(0, 5);

    return { nodes, edges, risk, ts, regime, forecast, topAnomalies };
  }, [state]);

  return { state, ...derived, status };
}