// src/components/Timeline.jsx
import { useEffect, useMemo, useRef, useState } from "react";

function clamp01(x) {
  const v = Number(x);
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

function fmtTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return "";
  }
}

// Draw a simple line sparkline in SVG
function Sparkline({ values = [], height = 60 }) {
  const w = 320;
  const h = height;
  const n = values.length;
  if (n < 2) return <div className="text-xs text-muted">No history yet.</div>;

  const pts = values.map((v, i) => {
    const x = (i / (n - 1)) * (w - 2) + 1;
    const y = (1 - clamp01(v)) * (h - 2) + 1;
    return [x, y];
  });

  const d = pts
    .map(([x, y], i) => `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full"
      style={{ display: "block" }}
    >
      <path d={d} fill="none" stroke="rgba(34,197,94,0.85)" strokeWidth="2" />
      <path
        d={`${d} L ${w - 1} ${h - 1} L 1 ${h - 1} Z`}
        fill="rgba(34,197,94,0.12)"
      />
    </svg>
  );
}

export default function Timeline({ risk, historyFromApi }) {
  // Local rolling history fallback (last ~60 points)
  const [localHistory, setLocalHistory] = useState([]);
  const lastTsRef = useRef(null);

  useEffect(() => {
    const ts = Date.now();
    if (lastTsRef.current && ts - lastTsRef.current < 400) return;
    lastTsRef.current = ts;

    const pca = clamp01(risk?.pca ?? 0);
    setLocalHistory((prev) => {
      const next = [...prev, { ts, pca }];
      return next.slice(-60);
    });
  }, [risk?.pca]);

  const history = useMemo(() => {
    const h = Array.isArray(historyFromApi) && historyFromApi.length
      ? historyFromApi
      : localHistory;

    // Normalize to {ts, pca}
    return h
      .map((x) => ({
        ts: x.ts ?? x.t ?? Date.now(),
        pca: clamp01(x.pca ?? x.system ?? x.risk ?? 0),
      }))
      .slice(-60);
  }, [historyFromApi, localHistory]);

  const values = history.map((x) => x.pca);
  const latest = history[history.length - 1];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs text-muted">System Risk (PCA) – last ~60 ticks</div>
        <div className="text-xs text-muted">
          {latest ? fmtTime(latest.ts) : ""}
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-bg p-3">
        <Sparkline values={values} />
      </div>

      <div className="text-[11px] text-muted">
        Higher line = stronger common factor / higher contagion.
      </div>
    </div>
  );
}