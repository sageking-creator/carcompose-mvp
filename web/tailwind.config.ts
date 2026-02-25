import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#f1ecdf",
        ink: "#111111",
        accent: "#c94f00",
        panel: "#fff8ea"
      },
      fontFamily: {
        display: ["\"Space Grotesk\"", "ui-sans-serif", "system-ui", "sans-serif"],
        body: ["\"IBM Plex Sans\"", "ui-sans-serif", "system-ui", "sans-serif"]
      }
    }
  },
  plugins: []
};

export default config;
