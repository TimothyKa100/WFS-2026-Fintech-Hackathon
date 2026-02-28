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
        "bg-panel border border-border rounded-2xl shadow-soft overflow-hidden",
        className,
      ].join(" ")}
    >
      {(title || right) && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div className="text-sm font-semibold text-text tracking-wide">
            {title}
          </div>
          <div className="text-xs text-muted">{right}</div>
        </div>
      )}
      <div className={["p-4", bodyClassName].join(" ")}>{children}</div>
    </div>
  );
}