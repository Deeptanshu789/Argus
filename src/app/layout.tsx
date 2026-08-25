export const metadata = { title: "Argus", description: "City-wide ANPR and trajectory tracking" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#0b0f14", color: "#e6edf3",
                     fontFamily: "ui-sans-serif, system-ui, sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
