/**
 * Placeholder. Dev B replaces this with the dashboard shell and the four views.
 * It exists only so the app boots and the mock API can be poked from a browser.
 */
export default function Page() {
  const routes = [
    "/api/mock/cameras", "/api/mock/cameras/links", "/api/mock/tracks?limit=5",
    "/api/mock/trajectories?limit=3", "/api/mock/search?plate=KA05MR7821",
    "/api/mock/analytics", "/api/mock/alerts?acked=false",
  ];
  return (
    <main style={{ padding: "3rem", maxWidth: 720 }}>
      <h1 style={{ margin: 0 }}>Argus</h1>
      <p style={{ opacity: 0.7 }}>
        Mock API is live. WebSocket at <code>/ws</code>. Dashboard goes here.
      </p>
      <ul style={{ lineHeight: 1.9 }}>
        {routes.map((r) => <li key={r}><a href={r} style={{ color: "#6cb6ff" }}>{r}</a></li>)}
      </ul>
    </main>
  );
}
