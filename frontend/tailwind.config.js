/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0f14",
        panel: "#0f172a",
        panel2: "#111827",
        border: "#1f2937",
        text: "#e5e7eb",
        muted: "#9ca3af",
        accent: "#22c55e",
        danger: "#ef4444",
        warning: "#f59e0b",
      },
      boxShadow: {
        soft: "0 10px 30px rgba(0,0,0,0.35)",
      },
    },
  },
  plugins: [],
};