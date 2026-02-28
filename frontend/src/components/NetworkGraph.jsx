// src/components/NetworkGraph.jsx
import { useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const clamp01 = (x) => Math.max(0, Math.min(1, x));

/**
 * Renders a contagion network.
 * CHANGE: Only renders edges that are among the top-K (default 3)
 * most-correlated edges PER NODE (by |weight|).
 *
 * Notes:
 * - Works best if `edges` includes a dense set of pairwise correlations
 *   (or at least enough edges per node to choose top 3).
 * - If your backend already sends a sparse list, this will further sparsify it.
 */
export default function NetworkGraph({
  nodes = [],
  edges = [],
  height = 520,
  topK = 3,
  minAbsCorr = 0, // optional: set like 0.25 or 0.35 to avoid weak edges
}) {
  const fgRef = useRef(null);
  const [selected, setSelected] = useState(null);
  const [query, setQuery] = useState("");

  const q = query.trim().toLowerCase();

  const highlightedId = useMemo(() => {
    if (!q) return null;
    const hit = (nodes || []).find((n) =>
      String(n.label || n.id).toLowerCase().includes(q)
    );
    return hit?.id || null;
  }, [q, nodes]);

  const nodeById = useMemo(() => {
    const m = new Map();
    (nodes || []).forEach((n) => m.set(n.id, n));
    return m;
  }, [nodes]);

  function focusNodeById(id) {
    const n = nodeById.get(id);
    if (!n) return;
    setSelected(n);
    fgRef.current?.centerAt?.(n.x, n.y, 300);
    fgRef.current?.zoom?.(2.2, 350);
  }

  // --- Contagion "pulse" based on time. Works for mock + live ---
  const nowMs = Date.now();
  const pulse = useMemo(() => 0.5 + 0.5 * Math.sin(nowMs / 180), [nowMs]);

  // ---------- NEW: Top-K per node edge selection ----------
  const graphData = useMemo(() => {
    const safeNodes = nodes || [];

    // Normalize incoming edges while preserving extra fields
    const normalized = (edges || [])
      .map((e) => {
        const w = Number(e.weight ?? 0);
        const absW = Number(e.absWeight ?? Math.abs(w));
        return {
          ...e,
          source: e.source,
          target: e.target,
          weight: w,
          absWeight: absW,
          deltaAbsWeight: Number(e.deltaAbsWeight ?? 0),
          spiked: !!e.spiked,
        };
      })
      // optionally remove very weak edges
      .filter(
        (e) => Math.abs(Number(e.absWeight ?? Math.abs(e.weight))) >= minAbsCorr
      );

    // Build adjacency: for each nodeId, list incident edges with absWeight
    const adj = new Map();
    for (const e of normalized) {
      const a = typeof e.source === "object" ? e.source.id : e.source;
      const b = typeof e.target === "object" ? e.target.id : e.target;
      if (a == null || b == null) continue;

      if (!adj.has(a)) adj.set(a, []);
      if (!adj.has(b)) adj.set(b, []);

      // treat as undirected for ranking per node
      adj.get(a).push({ edge: e, other: b, absW: e.absWeight });
      adj.get(b).push({ edge: e, other: a, absW: e.absWeight });
    }

    // For each node, take topK incident edges by absW
    const pickedKeys = new Set();

    const edgeKey = (u, v) => {
      const su = String(u);
      const sv = String(v);
      return su < sv ? `${su}||${sv}` : `${sv}||${su}`;
    };

    for (const n of safeNodes) {
      const id = n?.id;
      if (id == null) continue;

      const list = adj.get(id) || [];
      if (!list.length) continue;

      // sort descending by abs corr
      list.sort((x, y) => (y.absW ?? 0) - (x.absW ?? 0));

      // take topK (or fewer)
      for (let i = 0; i < Math.min(topK, list.length); i++) {
        const other = list[i].other;
        pickedKeys.add(edgeKey(id, other));
      }
    }

    // Keep edge if it was selected by either endpoint (since we used undirected keys)
    const links = normalized
      .filter((e) => {
        const a = typeof e.source === "object" ? e.source.id : e.source;
        const b = typeof e.target === "object" ? e.target.id : e.target;
        if (a == null || b == null) return false;
        return pickedKeys.has(edgeKey(a, b));
      })
      .map((e) => ({
        // ForceGraph wants links with source/target ids
        source: typeof e.source === "object" ? e.source.id : e.source,
        target: typeof e.target === "object" ? e.target.id : e.target,
        weight: e.weight,
        absWeight: e.absWeight,
        deltaAbsWeight: e.deltaAbsWeight,
        spiked: e.spiked,
        // preserve any extra fields
        ...e,
      }));

    return { nodes: safeNodes, links };
  }, [nodes, edges, topK, minAbsCorr]);

  return (
    <div className="space-y-3">
      {/* CARD: graph + controls (controls are BELOW graph, not clipped) */}
      <div className="rounded-2xl border border-border bg-bg">
        {/* Only the graph area is overflow-hidden */}
        <div className="overflow-hidden rounded-2xl">
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            height={height}
            backgroundColor="#0b0f14"
            nodeRelSize={5}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const label = node.label || node.id;
              const r = 6;

              const group = node.group || "default";
              const fill = "rgba(59,130,246,0.95)"; // all blue

              // anomaly halo
              // anomaly indicator (subtle glow + thin ring)
              // much smaller anomaly ring
              const anomaly = clamp01(Number(node.anomaly ?? 0));

              if (anomaly > 0.02) {
                const ringR = r + 2 + 3 * anomaly; // WAS big, now tiny

                ctx.beginPath();
                ctx.arc(node.x, node.y, ringR, 0, 2 * Math.PI, false);
                ctx.strokeStyle = `rgba(250, 204, 21, ${0.2 + 0.4 * anomaly})`;
                ctx.lineWidth = (1 + 1 * anomaly) / globalScale; // thinner
                ctx.stroke();
              }

              // main node
              ctx.beginPath();
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
              ctx.fillStyle = fill;
              ctx.fill();

              // search highlight
              if (highlightedId && node.id === highlightedId) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, r + 6, 0, 2 * Math.PI, false);
                ctx.strokeStyle = "rgba(250,204,21,0.55)";
                ctx.lineWidth = 3 / globalScale;
                ctx.stroke();
              }

              // selected ring
              if (selected?.id === node.id) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI, false);
                ctx.strokeStyle = "rgba(34,197,94,0.35)";
                ctx.lineWidth = 3 / globalScale;
                ctx.stroke();
              }

              // label (hide when zoomed out)
              const fontSize = 12 / globalScale;
              if (fontSize > 2.5) {
                ctx.font = `${fontSize}px sans-serif`;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillStyle = "rgba(229,231,235,0.9)";
                ctx.fillText(label, node.x + 10, node.y);
              }
            }}
            linkWidth={(link) => {
              const w = Math.abs(Number(link.weight || 0));
              const base = 0.5 + 3 * w;
              const boosted = link.spiked ? base + 2.0 * pulse : base;
              return boosted;
            }}
            linkColor={(link) => {
              const w = Number(link.weight || 0);
              const absW = Math.abs(w);

              const baseAlpha = 0.12 + 0.55 * absW;
              const spikeBoost = link.spiked ? 0.25 * pulse : 0;

              const alpha = clamp01(baseAlpha + spikeBoost);

              return w >= 0
                ? `rgba(34,197,94,${alpha})`
                : `rgba(239,68,68,${alpha})`;
            }}
            linkLineDash={(link) =>
              Number(link.weight || 0) < 0 ? [4, 4] : null
            }
            linkCurvature={0.05}
            onNodeClick={(node) => {
              setSelected(node);
              fgRef.current?.centerAt?.(node.x, node.y, 300);
              fgRef.current?.zoom?.(2.2, 350);
            }}
            onNodeHover={(node) => {
              const el = document.querySelector("canvas");
              if (el) el.style.cursor = node ? "pointer" : "default";
            }}
            nodeLabel={(node) => {
              const n = nodeById.get(node.id) || node;
              const ret =
                n.ret != null ? (Number(n.ret) * 100).toFixed(2) : "—";
              const vol =
                n.vol != null ? (Number(n.vol) * 100).toFixed(1) : "—";
              const px = n.price != null ? Number(n.price).toFixed(2) : "—";
              const an =
                n.anomaly != null
                  ? (Number(n.anomaly) * 100).toFixed(0)
                  : "—";
              return `${n.label || n.id}\nPrice: ${px}\nRet: ${ret}%\nVol: ${vol}%\nAnomaly: ${an}%`;
            }}
          />
        </div>

        {/* CONTROLS: revamped visual (same handlers/state, just styling/layout) */}
        <div className="p-3 border-t border-border">
          <div className="rounded-2xl border border-border bg-panel2/70 backdrop-blur-sm shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
            <div className="flex flex-wrap items-center justify-between gap-3 p-3">
              {/* Left: Search + Focus */}
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-border bg-bg/50 text-[11px] text-muted">
                    <span className="h-1.5 w-1.5 rounded-full bg-amber-300" />
                    Search & Focus
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <div className="relative">
                    <input
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="BTC / ETH / SOL…"
                      className={[
                        "w-64 bg-bg border border-border rounded-2xl pl-4 pr-24 py-2.5 text-sm outline-none",
                        "focus:border-muted/60 focus:ring-2 focus:ring-amber-300/10",
                        "placeholder:text-muted/60",
                      ].join(" ")}
                    />
                    <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2">
                      <span className="hidden sm:inline text-[10px] px-2 py-1 rounded-full border border-border bg-bg/60 text-muted">
                        {highlightedId ? "match" : "no match"}
                      </span>
                      <button
                        className={[
                          "text-xs px-3 py-2 rounded-xl border border-border bg-bg",
                          "hover:border-muted/40 transition",
                          "disabled:opacity-50 disabled:cursor-not-allowed",
                        ].join(" ")}
                        onClick={() => {
                          if (highlightedId) focusNodeById(highlightedId);
                        }}
                        disabled={!highlightedId}
                      >
                        Focus
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right: Reset */}
              <div className="flex items-center gap-2">
                <span className="hidden sm:inline text-[11px] text-muted">
                  Tip: click a node to inspect
                </span>
                <button
                  className={[
                    "text-xs px-3 py-2 rounded-xl border border-border bg-bg",
                    "hover:border-muted/40 transition",
                  ].join(" ")}
                  onClick={() => {
                    setSelected(null);
                    fgRef.current?.zoomToFit?.(350, 60);
                  }}
                >
                  Reset view
                </button>
              </div>
            </div>

            {/* Subtle divider + status strip */}
            <div className="border-t border-border/80 px-3 py-2 flex flex-wrap items-center justify-between gap-2">
              <div className="text-[11px] text-muted flex items-center gap-2">
                <span className="inline-flex items-center gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-slate-300/80" />
                  Rendering edges: top {topK} per asset
                </span>
                {minAbsCorr > 0 ? (
                  <span className="px-2 py-0.5 rounded-full border border-border bg-bg/50">
                    |corr| ≥ {minAbsCorr}
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded-full border border-border bg-bg/50">
                    no minimum filter
                  </span>
                )}
              </div>

              <div className="text-[11px] text-muted flex items-center gap-2">
                <span className="px-2 py-0.5 rounded-full border border-border bg-bg/50">
                  Nodes: {(nodes || []).length}
                </span>
                <span className="px-2 py-0.5 rounded-full border border-border bg-bg/50">
                  Links: {(graphData?.links || []).length}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Selected node panel (revamped visual only) */}
      <div className="rounded-2xl border border-border bg-panel2 p-3">
        {selected ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-2xl border border-border bg-bg/60 flex items-center justify-center shadow-sm">
                <div className="h-3 w-3 rounded-full bg-green-400/80" />
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wider text-muted">
                  Selected asset
                </div>
                <div className="font-semibold text-text text-base leading-tight">
                  {selected.label || selected.id}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="px-2.5 py-1 rounded-full border border-border bg-bg/50 text-xs text-muted">
                price{" "}
                <span className="text-text font-semibold">
                  {selected.price != null ? Number(selected.price).toFixed(2) : "—"}
                </span>
              </span>

              <span className="px-2.5 py-1 rounded-full border border-border bg-bg/50 text-xs text-muted">
                ret{" "}
                <span className="text-text font-semibold">
                  {selected.ret != null
                    ? `${(Number(selected.ret) * 100).toFixed(2)}%`
                    : "—"}
                </span>
              </span>

              <span className="px-2.5 py-1 rounded-full border border-border bg-bg/50 text-xs text-muted">
                vol{" "}
                <span className="text-text font-semibold">
                  {selected.vol != null
                    ? `${(Number(selected.vol) * 100).toFixed(1)}%`
                    : "—"}
                </span>
              </span>

              <span className="px-2.5 py-1 rounded-full border border-border bg-bg/50 text-xs text-muted">
                anomaly{" "}
                <span className="text-text font-semibold">
                  {selected.anomaly != null
                    ? `${(Number(selected.anomaly) * 100).toFixed(0)}%`
                    : "—"}
                </span>
              </span>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-2xl border border-border bg-bg/60 flex items-center justify-center">
                <div className="h-2.5 w-2.5 rounded-full bg-slate-300/70" />
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wider text-muted">
                  Inspector
                </div>
                <div className="text-muted text-sm">
                  No node selected. Click an asset to inspect.
                </div>
              </div>
            </div>
            <div className="hidden sm:flex text-[11px] text-muted items-center gap-2">
              <span className="px-2 py-1 rounded-full border border-border bg-bg/50">
                drag to pan
              </span>
              <span className="px-2 py-1 rounded-full border border-border bg-bg/50">
                scroll to zoom
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Legend (revamped visual only) */}
      <div className="rounded-2xl border border-border bg-panel2/60 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <div className="text-[11px] uppercase tracking-wider text-muted">
            Legend
          </div>
          <div className="text-[11px] text-muted">
            Visual encoding of correlations + signals
          </div>
        </div>

        <div className="flex flex-wrap gap-2 text-xs">
          <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-border bg-bg/50 text-muted">
            <span className="h-2 w-2 rounded-full bg-green-400" />
            Positive corr
          </span>

          <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-border bg-bg/50 text-muted">
            <span className="h-2 w-2 rounded-full bg-red-400" />
            Negative corr (dashed)
          </span>

          <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-border bg-bg/50 text-muted">
            <span className="h-2 w-2 rounded-full bg-amber-300" />
            Anomaly halo
          </span>

          <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-border bg-bg/50 text-muted">
            <span className="h-2 w-2 rounded-full bg-amber-200" />
            Correlation spike (pulse)
          </span>

          <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-border bg-bg/50 text-muted">
            <span className="h-2 w-2 rounded-full bg-amber-300" />
            Search highlight
          </span>

          <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-border bg-bg/50 text-muted">
            <span className="h-2 w-2 rounded-full bg-slate-300" />
            Edges: top {topK} per asset
            {minAbsCorr > 0 ? ` (|corr| ≥ ${minAbsCorr})` : ""}
          </span>
        </div>
      </div>
    </div>
  );
}