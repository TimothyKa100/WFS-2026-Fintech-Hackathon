// src/components/layout/Panel.jsx
export default function Panel({
  title,
  right,
  children,
  className = "",
  bodyClassName = "",
}) {
  return (
    <div
      className={[
        // Base
        "bg-panel border border-border rounded-2xl shadow-soft overflow-hidden",
        // Make it feel premium
        "backdrop-blur-sm transition-transform duration-200 will-change-transform",
        "hover:-translate-y-[1px]",
        className,
      ].join(" ")}
    >
      {(title || right) && (
        <div
          className={[
            "flex items-center justify-between",
            "px-4 py-3 border-b border-border",
            // subtle header treatment
            "bg-bg/20",
          ].join(" ")}
        >
          <div className="flex items-center gap-2 min-w-0">
            {/* Tiny accent dot (purely visual) */}
            <span className="h-2 w-2 rounded-full bg-white/20 shrink-0" />
            <div className="text-sm font-semibold text-text tracking-wide truncate">
              {title}
            </div>
          </div>

          <div className="text-[11px] text-muted shrink-0">
            {right ? (
              <span className="px-2 py-0.5 rounded-full border border-border bg-bg/30">
                {right}
              </span>
            ) : null}
          </div>
        </div>
      )}

      <div
        className={[
          // slightly better breathing room than p-4, but still consistent
          "p-5",
          bodyClassName,
        ].join(" ")}
      >
        {children}
      </div>
    </div>
  );
}