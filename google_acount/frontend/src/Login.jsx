import { useEffect, useRef } from "react";

export default function Login({ onLogin }) {
  const buttonRef = useRef(null);

  useEffect(() => {
    window.google.accounts.id.initialize({
      client_id: import.meta.env.VITE_GOOGLE_CLIENT_ID,
      callback: async (response) => {
        // response.credential is the Google ID token (JWT)
        const res = await fetch("http://localhost:8000/auth/google", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id_token: response.credential }),
        });
        if (!res.ok) {
          alert("Login failed on the server");
          return;
        }
        const data = await res.json(); // { access_token, token_type, user }
        onLogin(data);
      },
    });

    window.google.accounts.id.renderButton(buttonRef.current, {
      theme: "outline",
      size: "large",
    });
  }, [onLogin]);

  return <div ref={buttonRef} style={{ display: "flex", justifyContent: "center" }} />;
}