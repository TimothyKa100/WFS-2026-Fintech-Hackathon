// src/api/transform.js

/**
 * Convert nested correlation dict:
 * {
 *   "Bitcoin": { "Bitcoin": 1, "Ethereum": 0.88, ... },
 *   "Ethereum": { "Bitcoin": 0.88, "Ethereum": 1, ... },
 *   ...
 * }
 *
 * Into:
 * {
 *   assets: string[],
 *   matrix: number[][]
 * }
 */
export function corrDictToMatrix(corrDict) {
  if (!corrDict || typeof corrDict !== "object") {
    return { assets: [], matrix: [] };
  }

  const assets = Object.keys(corrDict);

  const matrix = assets.map((row) =>
    assets.map((col) => {
      const value = corrDict?.[row]?.[col];
      return typeof value === "number" ? value : null;
    })
  );

  return { assets, matrix };
}

/**
 * Alternative format for heatmap libraries that expect triples:
 * [
 *   { x: "Bitcoin", y: "Ethereum", value: 0.88 },
 *   ...
 * ]
 */
export function corrDictToTriples(corrDict) {
  if (!corrDict || typeof corrDict !== "object") {
    return { assets: [], data: [] };
  }

  const assets = Object.keys(corrDict);
  const data = [];

  for (const row of assets) {
    for (const col of assets) {
      const value = corrDict?.[row]?.[col];
      if (typeof value === "number") {
        data.push({
          x: col,
          y: row,
          value,
        });
      }
    }
  }

  return { assets, data };
}