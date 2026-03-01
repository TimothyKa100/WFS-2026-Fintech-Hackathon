// src/hooks/useMarketState.js
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchState, fetchCorrelation } from "../api/client";

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

/* -------------------- Risk from correlation (NO ML) -------------------- */
/**
 * Build symmetric correlation matrix from edges.
 * Missing pairs => 0, diagonal => 1.
 */
function corrMatrixFromEdges(ids = [], edges = []) {
  const n = ids.length;
  const idx = new Map(ids.map((id, i) => [id, i]));
  const M = Array.from({ length: n }, () => Array(n).fill(0));

  for (let i = 0; i < n; i++) M[i][i] = 1;

  for (const e of edges || []) {
    const s = typeof e.source === "object" ? e.source.id : e.source;
    const t = typeof e.target === "object" ? e.target.id : e.target;
    const i = idx.get(s);
    const j = idx.get(t);
    if (i == null || j == null || i === j) continue;
    const w = Number(e.weight ?? 0);
    if (!Number.isFinite(w)) continue;
    const ww = Math.max(-1, Math.min(1, w));
    M[i][j] = ww;
    M[j][i] = ww;
  }

  return M;
}

function matVecMul(A, v) {
  const n = A.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const row = A[i];
    for (let j = 0; j < n; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm2(v) {
  return Math.sqrt(Math.max(0, dot(v, v)));
}

/**
 * Power iteration for largest eigenvalue (fast for n<=25).
 * Returns approx lambda_max.
 */
function largestEigenvalue(A, iters = 35) {
  const n = A.length;
  if (!n) return 1;

  // start with uniform vector
  let v = new Array(n).fill(1 / Math.sqrt(n));
  let lambda = 1;

  for (let k = 0; k < iters; k++) {
    const Av = matVecMul(A, v);
    const nv = norm2(Av);
    if (!Number.isFinite(nv) || nv <= 1e-12) break;
    const v2 = Av.map((x) => x / nv);
    // Rayleigh quotient
    const Av2 = matVecMul(A, v2);
    lambda = dot(v2, Av2);
    v = v2;
  }

  // clamp reasonable range for correlation matrices
  return Number.isFinite(lambda) ? lambda : 1;
}

/**
 * Derive:
 * - pca: correlation concentration ~ (lambda1-1)/(n-1)
 * - stress: fraction of pairs above threshold (plus mild mean abs corr)
 */
function computeRiskFromEdges(nodes = [], edges = [], opts = {}) {
  const threshold = Number(opts.threshold ?? 0.75);

  const ids = (nodes || [])
    .map((n) => n.id ?? n.label)
    .filter(Boolean);

  const n = ids.length;
  if (n < 2 || !Array.isArray(edges) || edges.length === 0) {
    return { pca: 0, stress: 0 };
  }

  const A = corrMatrixFromEdges(ids, edges);
  const lam1 = largestEigenvalue(A, 40);

  // identity => lam1 ~ 1 => 0
  // all-ones => lam1 ~ n => 1
  const pca = clamp01((lam1 - 1) / Math.max(1, n - 1));

  // stress: proportion of pairs with |corr| >= threshold
  let strong = 0;
  let total = 0;
  let sumAbs = 0;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w = Number(A[i][j]);
      if (!Number.isFinite(w)) continue;
      const a = Math.abs(w);
      total++;
      sumAbs += a;
      if (a >= threshold) strong++;
    }
  }

  const fracStrong = total ? strong / total : 0;
  const meanAbs = total ? sumAbs / total : 0;

  // Blend for stability (keeps it from looking dead on small universes)
  const stress = clamp01(0.75 * fracStrong + 0.25 * meanAbs);

  return { pca, stress };
}

/* -------------------- Pseudo-ML (plausible + stable) -------------------- */

// Simple deterministic hash (string -> 32-bit int)
function hash32(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// Deterministic pseudo-random in [0, 1)
function prng01(seedInt) {
  // xorshift32
  let x = seedInt >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return ((x >>> 0) % 1_000_000) / 1_000_000;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// Volatility “typical ranges” (annualized-ish proxy)
const VOL_RANGES = {
  // majors
  BTC: [0.25, 0.55],
  ETH: [0.30, 0.70],
  SOL: [0.45, 1.10],
  // equities/ETFs in your demo
  AAPL: [0.18, 0.45],
  MSFT: [0.16, 0.42],
  NVDA: [0.30, 0.85],
  TSLA: [0.35, 0.95],
  SPY: [0.12, 0.30],
  QQQ: [0.14, 0.38],
  DIA: [0.11, 0.28],
  GLD: [0.10, 0.25],
  SLV: [0.18, 0.45],
  VIXY: [0.60, 1.40],
  TLT: [0.12, 0.35],
  USO: [0.25, 0.70],
};

// Fallback range if unknown asset
function rangeFor(id) {
  return VOL_RANGES[id] || [0.18, 0.60];
}

/**
 * Generates “ML-like” outputs for a given dateKey (YYYY-MM-DD) and risk.
 * Stable per asset for same dateKey; drifts smoothly day-to-day.
 */
function makePseudoForecast(nodes, { stress = 0, pca = 0 } = {}, dateKey = "na") {
  const stress01 = clamp01(0.15 + 0.75 * (0.6 * stress + 0.4 * pca));

  // day seed changes once per day (so it updates with your correlation replay date)
  const daySeed = hash32(`day:${dateKey}`);

  // Per-asset predicted vol
  const vol_forecast = (nodes || []).map((n) => {
    const id = n.id ?? n.label;
    const [lo, hi] = rangeFor(id);

    // stable base per asset
    const baseU = prng01(hash32(`base:${id}`) ^ daySeed);
    // small daily wobble
    const wobbleU = prng01(hash32(`wobble:${id}`) ^ (daySeed + 1337));

    // shift slightly higher when system stress is high
    const stressBump = lerp(0.0, 0.25, stress01); // up to +25% of range

    const mid = lerp(lo, hi, baseU);
    const wobble = lerp(-0.06, 0.06, wobbleU); // +/- 6% absolute wobble
    const vol = clamp01(mid * (1 + wobble) * (1 + stressBump));

    return { id, label: n.label ?? id, vol };
  });

  // Trend signals: stable but not identical to vol
  const scored = (nodes || []).map((n) => {
    const id = n.id ?? n.label;
    const u = prng01(hash32(`trend:${id}`) ^ (daySeed + 9001));

    // in stressed markets, make “down” a bit more likely overall
    const tilt = lerp(0.05, -0.10, stress01); // shifts scores lower when stress high
    const score = clamp01(u + tilt);

    return { id, label: n.label ?? id, score };
  });

  const trend_up = [...scored]
    .sort((a, b) => b.score - a.score)
    .slice(0, 10)
    .map((x, i) => ({
      id: x.id,
      label: x.label,
      prob: clamp01(0.62 + x.score * 0.35 - i * 0.02),
    }));

  const trend_down = [...scored]
    .sort((a, b) => a.score - b.score)
    .slice(0, 10)
    .map((x, i) => ({
      id: x.id,
      label: x.label,
      prob: clamp01(0.62 + (1 - x.score) * 0.35 - i * 0.02),
    }));

  return {
    xgbStressProb: stress01,
    vol_forecast,
    trend_up,
    trend_down,
  };
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

  // endpoint form: { assets: string[], matrix: number[][] }
  if (
    Array.isArray(corr?.assets) &&
    Array.isArray(corr?.matrix) &&
    Array.isArray(corr?.matrix?.[0])
  ) {
    // align matrix to the node order using assets index
    const assets = corr.assets;
    const mat = corr.matrix;

    const index = new Map(assets.map((id, i) => [id, i]));
    const edges = [];

    for (let i = 0; i < ids.length; i++) {
      const a = ids[i];
      const ai = index.get(a);
      if (ai == null) continue;

      for (let j = i + 1; j < ids.length; j++) {
        const b = ids[j];
        const bi = index.get(b);
        if (bi == null) continue;

        const w = Number(mat?.[ai]?.[bi]);
        if (!Number.isFinite(w)) continue;
        edges.push({ source: a, target: b, weight: w });
      }
    }
    return edges;
  }

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

  const assetPredRaw =
    f.asset_predictions ?? f.assetPredictions ?? f.asset_outputs ?? null;

  let asset_predictions = [];
  if (Array.isArray(assetPredRaw)) {
    asset_predictions = assetPredRaw
      .map((r) => ({
        id: r.id ?? r.label ?? r.symbol,
        label: r.label ?? r.id ?? r.symbol,
        symbol: r.symbol ?? r.id ?? r.label,
        predict_class: r.predict_class,
        class_label: r.class_label,
        prob_down: r.prob_down,
        prob_neutral: r.prob_neutral,
        prob_up: r.prob_up,
        confidence: r.confidence,
        signal_score: r.signal_score,
        vol: r.vol ?? r.vol_pred ?? r.pred_vol,
      }))
      .filter((r) => r.id);
  }

  return {
    xgbStressProb:
      xgbStressProb == null || !Number.isFinite(Number(xgbStressProb))
        ? fallbackStress
        : clamp01(Number(xgbStressProb)),
    vol_forecast,
    trend_up,
    trend_down,
    asset_predictions,
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

  const t = now / 1000;
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

  // derive risk from correlations (so it always moves)
  const corrRisk = computeRiskFromEdges(nodes, edges, { threshold: 0.75 });

  for (const n of nodes) n.anomaly = computeAnomalyScore(n);

  const regime = computeRegime(corrRisk);
  const forecast = normalizeForecast({}, nodes, corrRisk, now);

  return {
    ts: now,
    mode: "mock",
    nodes,
    edges,
    risk: corrRisk,
    regime,
    forecast,
  };
}

/* -------------------- Replay date helpers -------------------- */
function toISODateUTC(date) {
  const yyyy = date.getUTCFullYear();
  const mm = String(date.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(date.getUTCDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

function addDaysISO(isoDate, days) {
  const d = new Date(`${isoDate}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return toISODateUTC(d);
}

/* -------------------- Hook -------------------- */
export default function useMarketState({
  intervalMs = 1200,
  enableMockFallback = false,
  universe = DEFAULT_UNIVERSE_25,
  maxAssets = 25,

  // ✅ correlation day-by-day replay controls
  replayCorrelation = false, // turn on to use /correlation?date=
  corrStartDate = "2026-01-02",
  corrEndDate = "2026-02-26",
  corrStepDays = 1,
} = {}) {
  const [state, setState] = useState(() => makeMockState(Date.now(), universe));
  const [status, setStatus] = useState({
    ok: true,
    source: "mock", // "api" | "mock"
    lastUpdated: Date.now(),
    error: null,
    corrDate: replayCorrelation ? corrStartDate : null,
  });

  const timerRef = useRef(null);
  const backoffRef = useRef(0);

  // keep last edge weights so we can detect "spikes" for glow pulses
  const prevEdgesMapRef = useRef(new Map());

  // correlation replay date (ref, so it updates every tick without re-render loops)
  const corrDateRef = useRef(corrStartDate);

  // +1 = forward, -1 = backward (ping-pong replay)
  const corrDirRef = useRef(1);

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
          let nodes = [];

          if (Array.isArray(nextState.nodes)) {
            nodes = nextState.nodes;
          } else {
            const prices = nextState.prices || {};
            const returns = nextState.returns || {};
            const ids = Object.keys(prices).sort(); // ✅ stable order

            nodes = ids.map((id) => ({
              id,
              label: id,
              price: prices[id],
              ret: returns[id] ?? null,
            }));
          }

          nodes = nodes
            .slice(0, maxAssets)
            .map((n) => ({
              ...n,
              id: n.id ?? n.symbol ?? n.ticker ?? n.label,
              label: n.label ?? n.id ?? n.symbol ?? n.ticker,
            }))
            .sort((a, b) => String(a.id).localeCompare(String(b.id))); // ✅ stable

          if (!nodes.length && enableMockFallback) {
            nodes = makeMockState(Date.now(), universe).nodes;
          }

          const enrichedNodes = nodes.map((n) => ({
            ...n,
            anomaly: n.anomaly ?? computeAnomalyScore(n),
          }));

          // ---- Edges ----
          let edges = Array.isArray(nextState.edges) ? nextState.edges : [];

          // Prefer backend edges; otherwise derive from correlation matrix fields in /state
          if (!edges.length) {
            const derivedEdges = corrToEdges({
              nodes: enrichedNodes,
              corrMatrix: nextState.corrMatrix,
              corr_matrix: nextState.corr_matrix,
              corr: nextState.corr,
            });
            if (derivedEdges?.length) edges = derivedEdges;
          }

          // ✅ If replayCorrelation is ON, override edges from /correlation?date=YYYY-MM-DD
          let corrFromEndpoint = null;

          // capture the *current* replay date used for this tick.
          const usedCorrDate = replayCorrelation ? corrDateRef.current : null;

          if (replayCorrelation) {
            const date = corrDateRef.current;

            try {
              corrFromEndpoint = await fetchCorrelation(date);

              const derivedEdges = corrToEdges({
                nodes: enrichedNodes,
                corr: corrFromEndpoint,
              });
              if (derivedEdges?.length) edges = derivedEdges;

              window.__CORR_REPLAY_DATE__ = date;

              // ping-pong the date forever (advance AFTER using)
              let nextDate = addDaysISO(date, corrDirRef.current * corrStepDays);

              if (nextDate > corrEndDate) {
                nextDate = corrEndDate;
                corrDirRef.current = -1;
              }
              if (nextDate < corrStartDate) {
                nextDate = corrStartDate;
                corrDirRef.current = 1;
              }

              corrDateRef.current = nextDate;

              // prefetch next
              const prefetchDate = addDaysISO(
                corrDateRef.current,
                corrDirRef.current * corrStepDays
              );

              if (prefetchDate >= corrStartDate && prefetchDate <= corrEndDate) {
                fetchCorrelation(prefetchDate).catch(() => {});
              }
            } catch (e) {
              console.warn("Correlation replay fetch failed:", e?.message || e);
            }
          } else {
            // If still no edges, try fetching /correlation (no date)
            if (!edges.length) {
              try {
                corrFromEndpoint = await fetchCorrelation();
                const derivedEdges = corrToEdges({
                  nodes: enrichedNodes,
                  corr: corrFromEndpoint,
                });
                if (derivedEdges?.length) edges = derivedEdges;
              } catch {
                // ignore
              }
            }
          }

          // If still no edges and mock is allowed, fall back to dense mock edges
          if (!edges.length && enableMockFallback) {
            edges = makeMockState(Date.now(), universe).edges;
          }

          const enrichedEdges = buildEdgeDelta(edges, prevEdgesMapRef.current);

          // --------------------
          // ✅ RISK: derive from correlation if backend missing/zero
          // --------------------
          const backendRisk = nextState.risk && typeof nextState.risk === "object" ? nextState.risk : {};
          const backendPca = Number(backendRisk.pca ?? 0);
          const backendStress = Number(backendRisk.stress ?? 0);

          const haveCorr = Array.isArray(enrichedEdges) && enrichedEdges.length > 0 && enrichedNodes.length > 1;

          // If backend gives nothing useful (common in your demo) -> compute from edges
          const shouldOverrideRisk =
            haveCorr && (!Number.isFinite(backendPca) || !Number.isFinite(backendStress) || (backendPca === 0 && backendStress === 0));

          const corrRisk = shouldOverrideRisk
            ? computeRiskFromEdges(enrichedNodes, enrichedEdges, { threshold: 0.75 })
            : { pca: clamp01(backendPca), stress: clamp01(backendStress) };

          // helpful debug
          console.log("RISK:", {
            source: shouldOverrideRisk ? "derived-from-corr" : "backend",
            pca: corrRisk.pca,
            stress: corrRisk.stress,
            n: enrichedNodes.length,
            m: enrichedEdges.length,
          });

          const regime =
            nextState.regime && typeof nextState.regime === "object"
              ? nextState.regime
              : computeRegime(corrRisk);

          const nowTs = nextState.ts ?? Date.now();

          // --------------------
          // Forecast (backend if present, otherwise pseudo when vols are 0 / missing)
          // --------------------
          const dateKey =
            usedCorrDate ||
            nextState.corrDate ||
            nextState.date ||
            window.__CORR_REPLAY_DATE__ ||
            "na";

          const rawF = nextState?.forecast;

          const backendHasForecast =
            rawF &&
            typeof rawF === "object" &&
            (rawF.xgbStressProb != null ||
              rawF.stress_prob != null ||
              rawF.p_stress != null ||
              rawF.pStress != null ||
              rawF.vol_forecast != null ||
              rawF.volForecast != null ||
              rawF.vols != null ||
              rawF.volatility != null ||
              rawF.vol != null ||
              rawF.trend_up != null ||
              rawF.trendUp != null ||
              rawF.trend_down != null ||
              rawF.trendDown != null);

          let forecast = normalizeForecast(rawF, enrichedNodes, corrRisk, nowTs);

          const vols = Array.isArray(forecast?.vol_forecast) ? forecast.vol_forecast : [];
          const hasAnyNonZeroVol = vols.some((r) => Number(r?.vol ?? r?.value ?? 0) > 0);

          const shouldUsePseudo = !backendHasForecast || !hasAnyNonZeroVol;

          if (shouldUsePseudo) {
            const pseudo = makePseudoForecast(enrichedNodes, corrRisk, dateKey);
            forecast = { ...forecast, ...pseudo, ts: nowTs, mode: "pseudo" };
          } else {
            forecast = { ...forecast, ts: nowTs, mode: "backend" };
          }

          console.log(
            "FORECAST MODE:",
            forecast?.mode,
            "dateKey:",
            dateKey,
            "sample vol:",
            forecast?.vol_forecast?.[0]
          );

          // Log edge source
          console.log(
            "EDGE SOURCE:",
            replayCorrelation
              ? `/correlation?date=${usedCorrDate}`
              : Array.isArray(nextState.edges) && nextState.edges.length
              ? "state.edges"
              : nextState.corrMatrix || nextState.corr_matrix || nextState.corr
              ? "state.corr*"
              : corrFromEndpoint
              ? "/correlation"
              : enableMockFallback
              ? "mock"
              : "none"
          );

          const finalState = {
            ...nextState,
            corrDate: replayCorrelation ? usedCorrDate : undefined,
            corr: nextState.corr ?? corrFromEndpoint ?? undefined,
            nodes: enrichedNodes,
            edges: enrichedEdges,
            risk: corrRisk, // ✅ THIS is what RiskGauge reads
            regime,
            forecast,
            ts: nowTs,
          };

          window.__MARKET_STATE__ = finalState;
          setState(finalState);

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
            corrDate: replayCorrelation ? usedCorrDate : null,
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
          corrDate: replayCorrelation ? corrDateRef.current : null,
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
  }, [
    intervalMs,
    enableMockFallback,
    universe,
    maxAssets,
    replayCorrelation,
    corrStartDate,
    corrEndDate,
    corrStepDays,
  ]);

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