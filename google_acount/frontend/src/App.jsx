import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import Login from "./Login";
import Items from "./Items";
import Profile from "./Profile";

export default function App() {
  const [auth, setAuth] = useState(() =>
    JSON.parse(localStorage.getItem("auth") || "null")
  );
  const [me, setMe] = useState(null);
  const [dash, setDash] = useState(null);
  const [section, setSection] = useState("dashboard");

  const token = auth?.access_token;

  // Load protected data (members only) whenever auth changes
  const loadProtected = useCallback(async () => {
    if (!token) return;
    try {
      const [meData, dashData] = await Promise.all([
        api("/me", { token }),
        api("/dashboard", { token }),
      ]);
      setMe(meData);
      setDash(dashData);
    } catch {
      logout(); // token invalid/expired → back to login
    }
  }, [token]);

  useEffect(() => {
    loadProtected();
  }, [loadProtected]);

  function handleLogin(data) {
    localStorage.setItem("auth", JSON.stringify(data));
    setAuth(data);
  }

  function logout() {
    localStorage.removeItem("auth");
    setAuth(null);
    setMe(null);
    setDash(null);
  }

  if (!auth) {
    return (
      <div style={{ maxWidth: 400, margin: "100px auto", textAlign: "center" }}>
        <h1>Sign in</h1>
        <Login onLogin={handleLogin} />
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 520, margin: "40px auto" }}>
      <header style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {me?.picture && (
          <img src={me.picture} alt="avatar" width={40} style={{ borderRadius: "50%" }} />
        )}
        <div style={{ flex: 1 }}>
          <strong>{me?.name || auth.user.name}</strong>
          <div style={{ fontSize: 14, color: "#666" }}>{me?.email}</div>
        </div>
        <button onClick={logout}>Log out</button>
      </header>

      <nav style={{ display: "flex", gap: 8, margin: "20px 0" }}>
        {["dashboard", "items", "profile"].map((s) => (
          <button
            key={s}
            onClick={() => setSection(s)}
            style={{ fontWeight: section === s ? "bold" : "normal" }}
          >
            {s[0].toUpperCase() + s.slice(1)}
          </button>
        ))}
      </nav>

      {section === "dashboard" && dash && (
        <section style={{ textAlign: "left" }}>
          <h2>{dash.message}</h2>
          <p>Member since: {dash.member_since ? new Date(dash.member_since).toLocaleDateString() : "—"}</p>
          <p>
            📦 Items: {dash.stats.item_count} · ✅ Completed: {dash.stats.items_completed}
          </p>
        </section>
      )}

      {section === "items" && <Items token={token} />}
      {section === "profile" && (
        <Profile token={token} user={me || auth.user} onSaved={setMe} />
      )}
    </div>
  );
}