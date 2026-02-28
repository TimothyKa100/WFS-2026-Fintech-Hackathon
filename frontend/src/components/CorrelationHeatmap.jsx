// src/components/CorrelationHeatmap.jsx
function clamp(x, a, b) {
  const v = Number(x);
  if (!Number.isFinite(v)) return 0;
  return Math.max(a, Math.min(b, v));
}

// Map [-1, 1] to a color. Green = positive, Red = negative.
function cellStyle(v) {
  const x = clamp(v, -1, 1);
  const a = Math.abs(x);

  // alpha increases with magnitude
  const alpha = 0.08 + 0.55 * a;

  // positive -> green, negative -> red
  const bg =
    x >= 0
      ? `rgba(34,197,94,${alpha})`
      : `rgba(239,68,68,${alpha})`;

  return { backgroundColor: bg };
}

export default function CorrelationHeatmap({ corrMatrix }) {
  const assets = corrMatrix?.assets || [];
  const matrix = corrMatrix?.matrix || [];

  const ok =
    Array.isArray(assets) &&
    assets.length > 1 &&
    Array.isArray(matrix) &&
    matrix.length === assets.length;

  if (!ok) {
    return (
      <div className="text-xs text-muted">
        No correlation matrix yet (backend can return{" "}
        <span className="text-text font-semibold">corr_matrix</span>).
      </div>
    );
  }

  const n = assets.length;

  return (
    <div className="space-y-3">
      <div className="text-xs text-muted">
        Correlation heatmap (rolling window)
      </div>

      <div className="overflow-auto rounded-2xl border border-border bg-bg">
        <div
          className="grid"
          style={{
            gridTemplateColumns: `140px repeat(${n}, 44px)`,
            minWidth: 140 + n * 44,
          }}
        >
          {/* top-left corner */}
          <div className="sticky left-0 top-0 z-10 bg-bg border-b border-border px-3 py-2 text-xs text-muted">
            Asset
          </div>

          {/* column headers */}
          {assets.map((a) => (
            <div
              key={`col-${a}`}
              className="sticky top-0 z-10 bg-bg border-b border-border text-[11px] text-muted px-2 py-2 text-center"
            >
              {a}
            </div>
          ))}

          {/* rows */}
          {assets.map((rowA, i) => (
            <Row
              key={`row-${rowA}`}
              rowLabel={rowA}
              row={matrix[i]}
              assets={assets}
              rowIndex={i}
            />
          ))}
        </div>
      </div>

      <div className="flex flex-wrap gap-3 text-[11px] text-muted">
        <span className="inline-flex items-center gap-2">
          <span className="h-2 w-2 rounded-sm bg-green-400/60" />
          + corr
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-2 w-2 rounded-sm bg-red-400/60" />
          − corr
        </span>
      </div>
    </div>
  );
}

function Row({ rowLabel, row = [], assets = [], rowIndex }) {
  return (
    <>
      {/* row header */}
      <div className="sticky left-0 z-10 bg-bg border-b border-border px-3 py-2 text-xs text-text">
        {rowLabel}
      </div>

      {/* cells */}
      {assets.map((colA, j) => {
        const v = Array.isArray(row) ? row[j] : 0;
        const diag = rowIndex === j;

        return (
          <div
            key={`${rowLabel}-${colA}`}
            className={[
              "border-b border-border text-[11px] px-1 py-2 text-center",
              diag ? "bg-panel2" : "",
            ].join(" ")}
            style={diag ? undefined : cellStyle(v)}
            title={`${rowLabel} vs ${colA}: ${(Number(v) || 0).toFixed(2)}`}
          >
            {diag ? "—" : (Number(v) || 0).toFixed(2)}
          </div>
        );
      })}
    </>
  );
}