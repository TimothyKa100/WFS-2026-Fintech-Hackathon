// src/api/client.js

const API_BASE =
  (import.meta.env.VITE_API_BASE || "http://localhost:8000").replace(/\/$/, "");

async function request(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} (${path}) ${text}`);
  }

  // Some endpoints may return empty bodies
  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return null;

  return res.json();
}

/**
 * Expected response shape (recommendation):
 * {
 *   ts: number,
 *   mode: "live" | "replay",
 *   nodes: [{ id, label, group?, vol?, price?, ret? }],
 *   edges: [{ source, target, weight, direction? }],
 *   risk: { pca?: number, stress?: number },
 *   history?: { ts: number, pca?: number, stress?: number }[],
 *   corr_matrix?: { assets: string[], matrix: number[][] }
 * }
 */
export function fetchState() {
  return request("/state");
}

/**
 * Controls payload example:
 * { mode, playing, speed, threshold, window, assets }
 */
export function postControls(payload) {
  return request("/controls", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export { API_BASE };