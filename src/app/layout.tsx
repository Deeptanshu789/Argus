import "./globals.css";

export const metadata = { title: "Argus", description: "City-wide ANPR and trajectory tracking" };

/**
 * `data-theme` is stamped before React hydrates.
 *
 * Without this the first paint uses the light palette and the toggle swaps it
 * a frame later, which on a dark control-room screen is a white flash in the
 * operator's eyes. The script is tiny and inline on purpose: anything fetched
 * arrives too late to prevent the flash it exists to prevent.
 */
const THEME_BOOT = `
try {
  var t = localStorage.getItem("argus-theme");
  if (!t) t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", t);
} catch (e) {
  document.documentElement.setAttribute("data-theme", "dark");
}`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href={"https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700"
                + "&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap"}
        />
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOT }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
