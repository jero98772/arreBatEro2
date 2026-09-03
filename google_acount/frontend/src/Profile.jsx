import { useState } from "react";
import { api } from "./api";

export default function Profile({ token, user, onSaved }) {
  const [name, setName] = useState(user.name || "");
  const [bio, setBio] = useState(user.bio || "");
  const [msg, setMsg] = useState("");

  async function save(e) {
    e.preventDefault();
    const updated = await api("/profile", { method: "PUT", token, body: { name, bio } });
    onSaved(updated);
    setMsg("Saved ✅");
    setTimeout(() => setMsg(""), 2000);
  }

  return (
    <section style={{ textAlign: "left" }}>
      <h3>My Profile 🔒</h3>
      <form onSubmit={save} style={{ display: "grid", gap: 8 }}>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
        <textarea
          value={bio}
          onChange={(e) => setBio(e.target.value)}
          placeholder="Bio"
          rows={3}
        />
        <div>
          <button type="submit">Save</button> <span>{msg}</span>
        </div>
      </form>
    </section>
  );
}