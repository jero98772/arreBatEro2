import { useEffect, useState } from "react";
import { api } from "./api";

export default function Items({ token }) {
  const [items, setItems] = useState([]);
  const [title, setTitle] = useState("");
  const [error, setError] = useState("");

  async function load() {
    setItems(await api("/items", { token }));
  }
  useEffect(() => {
    load().catch((e) => setError(e.message));
  }, [token]);

  async function add(e) {
    e.preventDefault();
    if (!title.trim()) return;
    await api("/items", { method: "POST", token, body: { title } });
    setTitle("");
    load();
  }

  async function toggle(id) {
    await api(`/items/${id}/toggle`, { method: "PATCH", token });
    load();
  }

  async function remove(id) {
    await api(`/items/${id}`, { method: "DELETE", token });
    load();
  }

  return (
    <section style={{ textAlign: "left" }}>
      <h3>My Items 🔒</h3>
      {error && <p style={{ color: "red" }}>{error}</p>}
      <form onSubmit={add} style={{ display: "flex", gap: 8 }}>
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="New item..."
          style={{ flex: 1 }}
        />
        <button type="submit">Add</button>
      </form>
      <ul style={{ listStyle: "none", padding: 0 }}>
        {items.map((i) => (
          <li key={i.id} style={{ display: "flex", gap: 8, margin: "6px 0" }}>
            <label style={{ flex: 1, textDecoration: i.done ? "line-through" : "none" }}>
              <input type="checkbox" checked={i.done} onChange={() => toggle(i.id)} />{" "}
              {i.title}
            </label>
            <button onClick={() => remove(i.id)}>✕</button>
          </li>
        ))}
        {items.length === 0 && <li>No items yet — add one above.</li>}
      </ul>
    </section>
  );
}