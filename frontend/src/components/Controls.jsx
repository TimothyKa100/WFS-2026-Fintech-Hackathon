// src/components/Controls.jsx
import { useEffect, useMemo, useState } from "react";
import { postControls } from "../api/client";

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function Btn({ children, onClick, tone = "neutral", disabled = false }) {
  const tones = {
    neutral: "bg-panel2 border-border hover:border-muted/40 text-text",
    accent: "bg-accent/15 border-accent/30 hover:border-accent/60 text-text",
    danger: "bg-red-500/10 border-red-500/30 hover:border-red-500/60 text-text",
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={[
        "px-3 py-2 rounded-xl border text-sm transition disabled:opacity-50 disabled:cursor-not-allowed",
        tones[tone] || tones.neutral,
      ].join(" ")}
    >
      {children}
    </button>
  );
}

export default function Controls({
  initial = { threshold: 0.55, window: 30, mode: "live", playing: true, speed: 1 },
  onChange,
}) {
  const [threshold, setThreshold] = useState(initial.threshold ?? 0.55);
  const [window, setWindow] = useState(initial.window ?? 30);
  const [mode, setMode] = useState(initial.mode ?? "live"); // live | replay
  const [playing, setPlaying] = useState(initial.playing ?? true);
  const [speed, setSpeed] = useState(initial.speed ?? 1);

  const payload = useMemo(
    () => ({
      threshold: Number(threshold),
      window: Number(window),
      mode,
      playing,
      speed: Number(speed),
    }),
    [threshold, window, mode, playing, speed]
  );

  // notify parent
  useEffect(() => {
    onChange?.(payload);
  }, [payload, onChange]);

  async function push(partial) {
    try {
      await postControls({ ...payload, ...partial });
    } catch {
      // ignore; backend may not be running yet
    }
  }

  return (
    <div className="space-y-4">
      {/* Mode + playback */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-xl border border-border overflow-hidden">
          <button
            className={[
              "px-3 py-2 text-sm",
              mode === "live" ? "bg-accent/15" : "bg-panel2",
            ].join(" ")}
            onClick={() => {
              setMode("live");
              push({ mode: "live" });
            }}
          >
            LIVE
          </button>
          <button
            className={[
              "px-3 py-2 text-sm border-l border-border",
              mode === "replay" ? "bg-accent/15" : "bg-panel2",
            ].join(" ")}
            onClick={() => {
              setMode("replay");
              push({ mode: "replay" });
            }}
          >
            REPLAY
          </button>
        </div>

        <Btn
          tone="accent"
          onClick={() => {
            setPlaying((p) => !p);
            push({ playing: !playing });
          }}
        >
          {playing ? "Pause" : "Play"}
        </Btn>

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">Speed</span>
          <select
            className="bg-panel2 border border-border rounded-xl px-3 py-2 text-sm"
            value={speed}
            onChange={(e) => {
              const v = Number(e.target.value);
              setSpeed(v);
              push({ speed: v });
            }}
          >
            <option value={0.5}>0.5×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </div>
      </div>

      {/* Threshold */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs text-muted">Edge threshold (|corr|)</div>
          <div className="text-xs font-semibold text-text">
            {Number(threshold).toFixed(2)}
          </div>
        </div>
        <input
          type="range"
          min={0.1}
          max={0.95}
          step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(clamp(Number(e.target.value), 0.1, 0.95))}
          onMouseUp={() => push({ threshold })}
          onTouchEnd={() => push({ threshold })}
          className="w-full accent-accent"
        />
      </div>

      {/* Window */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs text-muted">Rolling window (intervals)</div>
          <div className="text-xs font-semibold text-text">{window}</div>
        </div>
        <input
          type="range"
          min={10}
          max={120}
          step={1}
          value={window}
          onChange={(e) => setWindow(clamp(Number(e.target.value), 10, 120))}
          onMouseUp={() => push({ window })}
          onTouchEnd={() => push({ window })}
          className="w-full accent-accent"
        />
        <div className="text-[11px] text-muted mt-1">
          Tip: 30–60 is stable; 10 is twitchy.
        </div>
      </div>
    </div>
  );
}