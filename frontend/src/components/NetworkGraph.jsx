// src/components/NetworkGraph.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const clamp01 = (x) => Math.max(0, Math.min(1, x));

// Persist layout across refreshes
const POS_KEY = "contagion_graph_positions_v2";

/**
 * Renders a contagion network.
 *
 * Behavior:
 * - Graph "expands" once (force sim runs) then freezes in place.
 * - Node positions are persisted to localStorage so refreshes don't re-layout/jump.
 * - Nodes remain fixed (via fx/fy) unless moved by the user.
 * - User can drag nodes; on drop, the node stays pinned where released (and is saved).
 * - Edges/data can keep updating; node positions are preserved.
 * - Only renders edges that are among the top-K most-correlated edges PER NODE (by |weight|).
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
  const [frozen, setFrozen] = useState(false);

  // Keep stable node objects across updates so x/y do not reset.
  // id -> nodeObject (mutated by ForceGraph / d3)
  const nodesMapRef = useRef(new Map());

  // Saved positions cache (localStorage mirror)
  const savedPosRef = useRef({});

  // Load saved positions once
  useEffect(() => {
    try {
      const raw = localStorage.getItem(POS_KEY);
      savedPosRef.current = raw ? JSON.parse(raw) : {};
    } catch {
      savedPosRef.current = {};
    }
  }, []);

  const saveAllPositions = () => {
    const gd = fgRef.current?.graphData?.();
    if (!gd?.nodes?.length) return;

    const out = {};
    for (const n of gd.nodes) {
      if (n?.id == null) continue;
      const x = n.fx ?? n.x;
      const y = n.fy ?? n.y;
      if (x == null || y == null) continue;
      out[n.id] = { x, y };
    }

    try {
      localStorage.setItem(POS_KEY, JSON.stringify(out));
      savedPosRef.current = out;
    } catch {}
  };

  const saveOnePosition = (node) => {
    if (!node?.id) return;
    const x = node.fx ?? node.x;
    const y = node.fy ?? node.y;
    if (x == null || y == null) return;

    const next = { ...(savedPosRef.current || {}), [node.id]: { x, y } };
    savedPosRef.current = next;

    try {
      localStorage.setItem(POS_KEY, JSON.stringify(next));
    } catch {}
  };

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
    const gd = fgRef.current?.graphData?.();
    const n =
      (gd?.nodes || []).find((x) => x?.id === id) ||
      nodeById.get(id) ||
      null;
    if (!n) return;

    setSelected(n);
    fgRef.current?.centerAt?.(n.x, n.y, 300);
    fgRef.current?.zoom?.(2.2, 350);
  }

  // --- Contagion "pulse" based on time. Works for mock + live ---
  // NOTE: even if edges/weights don't change, re-renders will redraw the canvas.
  const pulse = useMemo(() => 0.5 + 0.5 * Math.sin(Date.now() / 180), []);

  /**
   * Build graphData with:
   * 1) Stable nodes (preserve x/y/fx/fy across streaming updates)
   * 2) Restored positions from localStorage (no refresh jump)
   * 3) Top-K per node edge selection (STABLE + deterministic)
   */
  const graphData = useMemo(() => {
    const incomingNodes = nodes || [];
    const map = nodesMapRef.current;

    // Upsert/merge node fields WITHOUT overwriting simulation coords.
    for (const n of incomingNodes) {
      if (n?.id == null) continue;

      const saved = savedPosRef.current?.[n.id];
      const prev = map.get(n.id);

      if (prev) {
        Object.assign(prev, n);

        if (saved && (prev.fx == null || prev.fy == null)) {
          prev.fx = saved.x;
          prev.fy = saved.y;
          prev.x = prev.x ?? saved.x;
          prev.y = prev.y ?? saved.y;
        }

        if (frozen && prev.x != null && prev.y != null) {
          prev.fx = prev.fx ?? prev.x;
          prev.fy = prev.fy ?? prev.y;
        }
      } else {
        const nn = saved
          ? { ...n, fx: saved.x, fy: saved.y, x: saved.x, y: saved.y }
          : { ...n };
        map.set(n.id, nn);
      }
    }

    // Optional: remove nodes that no longer exist in incoming list
    const incomingIds = new Set(incomingNodes.map((n) => n?.id));
    for (const id of Array.from(map.keys())) {
      if (!incomingIds.has(id)) map.delete(id);
    }

    const safeNodes = Array.from(map.values());

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

    // ✅ Deterministic sort tie-breaker to stop topK flipping between renders
    const stableSortAdjList = (list) => {
      list.sort((x, y) => {
        const d = (y.absW ?? 0) - (x.absW ?? 0);
        if (d !== 0) return d;

        // tie 1: by "other" id
        const xo = String(x.other);
        const yo = String(y.other);
        if (xo !== yo) return xo.localeCompare(yo);

        // tie 2: by endpoints (source|target)
        const xs = String(
          typeof x.edge.source === "object" ? x.edge.source.id : x.edge.source
        );
        const xt = String(
          typeof x.edge.target === "object" ? x.edge.target.id : x.edge.target
        );
        const ys = String(
          typeof y.edge.source === "object" ? y.edge.source.id : y.edge.source
        );
        const yt = String(
          typeof y.edge.target === "object" ? y.edge.target.id : y.edge.target
        );

        return `${xs}|${xt}`.localeCompare(`${ys}|${yt}`);
      });
    };

    for (const n of safeNodes) {
      const id = n?.id;
      if (id == null) continue;

      const list = adj.get(id) || [];
      if (!list.length) continue;

      stableSortAdjList(list);

      for (let i = 0; i < Math.min(topK, list.length); i++) {
        const other = list[i].other;
        pickedKeys.add(edgeKey(id, other));
      }
    }

    // Keep edge if it was selected by either endpoint (undirected keys)
    const links = normalized
      .filter((e) => {
        const a = typeof e.source === "object" ? e.source.id : e.source;
        const b = typeof e.target === "object" ? e.target.id : e.target;
        if (a == null || b == null) return false;
        return pickedKeys.has(edgeKey(a, b));
      })
      .map((e) => {
        // ✅ Spread FIRST, then enforce source/target ids LAST
        const src = typeof e.source === "object" ? e.source.id : e.source;
        const tgt = typeof e.target === "object" ? e.target.id : e.target;

        return {
          ...e,
          source: src,
          target: tgt,
          weight: e.weight,
          absWeight: e.absWeight,
          deltaAbsWeight: e.deltaAbsWeight,
          spiked: e.spiked,
        };
      });

    return { nodes: safeNodes, links };
  }, [nodes, edges, topK, minAbsCorr, frozen]);

  /**
   * Freeze all nodes in place (fx/fy) and persist layout.
   */
  const freezeAllNodes = () => {
    const gd = fgRef.current?.graphData?.();
    if (!gd?.nodes?.length) return;

    gd.nodes.forEach((n) => {
      if (n?.x == null || n?.y == null) return;
      n.fx = n.fx ?? n.x;
      n.fy = n.fy ?? n.y;
    });

    saveAllPositions();
    setFrozen(true);
  };

  /**
   * If new nodes appear AFTER frozen, briefly reheat, then freeze+persist again.
   */
  useEffect(() => {
    if (!frozen) return;

    const gd = fgRef.current?.graphData?.();
    const hasUnplaced = (gd?.nodes || []).some(
      (n) => n?.x == null || n?.y == null
    );

    if (hasUnplaced) {
      fgRef.current?.resumeAnimation?.();
      const t = setTimeout(() => {
        freezeAllNodes();
        fgRef.current?.pauseAnimation?.();
      }, 700);
      return () => clearTimeout(t);
    }
  }, [graphData, frozen]);

  return (
    <div className="space-y-3">
      <div className="rounded-2xl border border-border bg-bg">
        <div className="overflow-hidden rounded-2xl">
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            height={height}
            backgroundColor="#0b0f14"
            nodeRelSize={5}
            linkCurvature={0.05}
            cooldownTicks={frozen ? 0 : 140}
            d3VelocityDecay={0.35}
            onEngineStop={() => {
              freezeAllNodes();
              fgRef.current?.pauseAnimation?.();
            }}
            enableNodeDrag={true}
            onNodeDrag={(node) => {
              node.fx = node.x;
              node.fy = node.y;
            }}
            onNodeDragEnd={(node) => {
              node.fx = node.x;
              node.fy = node.y;
              saveOnePosition(node);
            }}
            nodeCanvasObject={(node, ctx, globalScale) => {
              const label = node.label || node.id;
              const r = 6;
              const fill = "rgba(59,130,246,0.95)";

              const anomaly = clamp01(Number(node.anomaly ?? 0));

              if (anomaly > 0.02) {
                const ringR = r + 2 + 3 * anomaly;

                ctx.beginPath();
                ctx.arc(node.x, node.y, ringR, 0, 2 * Math.PI, false);
                ctx.strokeStyle = `rgba(250, 204, 21, ${0.2 + 0.4 * anomaly})`;
                ctx.lineWidth = (1 + 1 * anomaly) / globalScale;
                ctx.stroke();
              }

              ctx.beginPath();
              ctx.arc(node.x, node.y, r, 0, 2 * Math.PI, false);
              ctx.fillStyle = fill;
              ctx.fill();

              if (highlightedId && node.id === highlightedId) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, r + 6, 0, 2 * Math.PI, false);
                ctx.strokeStyle = "rgba(250,204,21,0.55)";
                ctx.lineWidth = 3 / globalScale;
                ctx.stroke();
              }

              if (selected?.id === node.id) {
                ctx.beginPath();
                ctx.arc(node.x, node.y, r + 4, 0, 2 * Math.PI, false);
                ctx.strokeStyle = "rgba(34,197,94,0.35)";
                ctx.lineWidth = 3 / globalScale;
                ctx.stroke();
              }

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

        {/* CONTROLS */}
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

            {/* Status strip */}
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
                <span className="px-2 py-0.5 rounded-full border border-border bg-bg/50">
                  Layout: {frozen ? "fixed (drag to move)" : "settling…"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Selected node panel */}
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
                  {selected.price != null
                    ? Number(selected.price).toFixed(2)
                    : "—"}
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

      {/* Legend */}
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

          <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full border border-border bg-bg/50 text-muted">
            <span className="h-2 w-2 rounded-full bg-slate-300" />
            Layout: {frozen ? "fixed (drag to move)" : "settling…"}
          </span>
        </div>
      </div>
    </div>
  );
}