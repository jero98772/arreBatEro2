"""
=============================================================
  WOMPI + FASTAPI — Complete Tutorial: Charge $1 USD (≈ $4,200 COP)
=============================================================

IMPORTANT — HOW AMOUNTS WORK IN WOMPI:
  • Wompi only works with Colombian Pesos (COP).
  • Amounts are always in CENTS (last 2 digits = cents).
  • $1 USD ≈ $4,200 COP → in Wompi cents = 420000
  • Example: $100 COP → send 10000 | $1,000 COP → send 100000

ABOUT TOKENS:
  Wompi requires TWO acceptance tokens per transaction:
  1. acceptance_token  → user accepted Terms & Conditions
  2. accept_personal_auth → user accepted Personal Data policy
  These tokens expire in about 30 minutes, so always fetch
  them fresh right before showing the payment form to the user.
"""

import hashlib
import os
import uuid
import asyncio
import httpx

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from typing import Optional

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — replace with your Wompi dashboard keys
# ─────────────────────────────────────────────────────────────
# Get your keys at: https://comercios.wompi.co
# For testing, use the SANDBOX keys (pub_test_... / prv_test_...)

WOMPI_PUBLIC_KEY = os.getenv("WOMPI_PUBLIC_KEY")
WOMPI_PRIVATE_KEY = os.getenv("WOMPI_PRIVATE_KEY")
WOMPI_INTEGRITY_SECRET = os.getenv("WOMPI_INTEGRITY_SECRET")

if not WOMPI_PUBLIC_KEY or not WOMPI_PRIVATE_KEY or not WOMPI_INTEGRITY_SECRET:
    raise RuntimeError(
        "Missing Wompi credentials. Copy .env.example to .env and fill in your keys."
    )

# Sandbox base URL (use https://production.wompi.co/v1 for production)
WOMPI_BASE_URL = os.getenv("WOMPI_BASE_URL", "https://sandbox.wompi.co/v1")

# ~$1 USD in Colombian Pesos cents (adjust rate as needed)
AMOUNT_IN_CENTS = int(os.getenv("AMOUNT_IN_CENTS", "420000"))  # $4,200 COP = ~$1 USD


# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Wompi Payment Tutorial",
    description="Charge ~$1 USD using the Wompi API",
    version="1.0.0",
)


# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────
class CardTokenRequest(BaseModel):
    number: str  # Card number, 13–19 digits
    cvc: str  # 3 or 4 digit CVC
    exp_month: str  # "MM" e.g. "08"
    exp_year: str  # "YY" e.g. "28"
    card_holder: str  # Full name on card


class PaymentRequest(BaseModel):
    customer_email: EmailStr
    card_token: str  # Token from /tokenize-card
    installments: int = 1  # Number of installments
    acceptance_token: str  # From /acceptance-tokens
    accept_personal_auth: str  # From /acceptance-tokens
    customer_ip: Optional[str] = "127.0.0.1"


class TransactionStatusRequest(BaseModel):
    transaction_id: str


# ─────────────────────────────────────────────────────────────
# HELPER: Generate integrity signature
# ─────────────────────────────────────────────────────────────
def generate_integrity_signature(
    reference: str, amount_in_cents: int, currency: str
) -> str:
    """
    Wompi requires a SHA-256 signature to prevent tampering.
    Formula: SHA256( reference + amount_in_cents + currency + integrity_secret )
    """
    raw = f"{reference}{amount_in_cents}{currency}{WOMPI_INTEGRITY_SECRET}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────
# STEP 1: Get acceptance tokens
# ─────────────────────────────────────────────────────────────
@app.get("/acceptance-tokens", summary="Step 1 — Fetch acceptance tokens")
async def get_acceptance_tokens():
    """
    WHEN TO CALL THIS:
      Call this endpoint EVERY TIME before showing the payment form to your user.
      Tokens expire (~30 min), so never cache them.

    WHAT TO DO WITH THE RESPONSE:
      1. Display the permalink URLs as clickable links / checkboxes in your UI.
      2. The user MUST check both boxes (confirming they read the PDFs).
      3. Send acceptance_token and accept_personal_auth in the payment request.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{WOMPI_BASE_URL}/merchants/{WOMPI_PUBLIC_KEY}")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.json())

        data = resp.json()["data"]
        acceptance = data["presigned_acceptance"]
        personal = data["presigned_personal_data_auth"]

        return {
            "instructions": "Show both permalink URLs to your user. They must accept both before paying.",
            "acceptance_token": {
                "token": acceptance["acceptance_token"],
                "permalink": acceptance["permalink"],  # Show this link to the user
                "type": acceptance["type"],
            },
            "personal_data_token": {
                "token": personal["acceptance_token"],
                "permalink": personal["permalink"],  # Show this link to the user
                "type": personal["type"],
            },
        }


# ─────────────────────────────────────────────────────────────
# STEP 2: Tokenize the card
# ─────────────────────────────────────────────────────────────
@app.post("/tokenize-card", summary="Step 2 — Tokenize a credit/debit card")
async def tokenize_card(card: CardTokenRequest):
    """
    WHEN TO CALL THIS:
      After the user fills the card form on your frontend.
      Ideally call this from the browser (using the public key) to avoid
      raw card data passing through your server. For this tutorial we do
      it server-side for simplicity.

    IMPORTANT:
      • Never store raw card numbers on your servers.
      • Each token is single-use. For recurring payments use Payment Sources.
      • The token expires in ~10 minutes if not used.

    SANDBOX TEST CARDS:
      Approved:  4242 4242 4242 4242 | CVC: any 3 digits | Exp: any future date
      Declined:  4111 1111 1111 1111
    """
    payload = {
        "number": card.number,
        "cvc": card.cvc,
        "exp_month": card.exp_month,
        "exp_year": card.exp_year,
        "card_holder": card.card_holder,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{WOMPI_BASE_URL}/tokens/cards",
            json=payload,
            headers={"Authorization": f"Bearer {WOMPI_PUBLIC_KEY}"},
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=resp.status_code, detail=resp.json())

        token_data = resp.json()["data"]
        return {
            "card_token": token_data["id"],  # ← use this in /pay
            "brand": token_data["brand"],
            "last_four": token_data["last_four"],
            "expires_at": token_data["expires_at"],
        }


# ─────────────────────────────────────────────────────────────
# STEP 3: Create the payment transaction
# ─────────────────────────────────────────────────────────────
@app.post("/pay", summary="Step 3 — Create a ~$1 USD payment transaction")
async def create_payment(payment: PaymentRequest):
    """
    AMOUNT: ~$1 USD = $4,200 COP = 420000 cents (hardcoded in this tutorial).

    FLOW AFTER THIS CALL:
      - Transaction is created with status: PENDING
      - Poll GET /transaction/{id} every 5 seconds
      - Wait for status: APPROVED | DECLINED | VOIDED | ERROR
    """
    # Generate a unique reference for this transaction
    reference = f"ORDER-{uuid.uuid4().hex[:12].upper()}"

    # Generate the integrity signature (server-side only — never expose the secret)
    signature = generate_integrity_signature(reference, AMOUNT_IN_CENTS, "COP")

    payload = {
        "amount_in_cents": AMOUNT_IN_CENTS,
        "currency": "COP",
        "customer_email": payment.customer_email,
        "reference": reference,
        "signature": signature,
        "acceptance_token": payment.acceptance_token,
        "accept_personal_auth": payment.accept_personal_auth,
        "ip": payment.customer_ip,
        "payment_method": {
            "type": "CARD",
            "token": payment.card_token,
            "installments": payment.installments,
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{WOMPI_BASE_URL}/transactions",
            json=payload,
            headers={"Authorization": f"Bearer {WOMPI_PRIVATE_KEY}"},
        )
        if resp.status_code not in (200, 201):
            raise HTTPException(status_code=resp.status_code, detail=resp.json())

        tx = resp.json()["data"]
        return {
            "transaction_id": tx["id"],
            "reference": tx["reference"],
            "status": tx["status"],  # Will be "PENDING" initially
            "amount_cop": AMOUNT_IN_CENTS / 100,
            "next_step": f"Poll GET /transaction/{tx['id']} until status is final",
        }


# ─────────────────────────────────────────────────────────────
# STEP 4: Check transaction status (polling)
# ─────────────────────────────────────────────────────────────
@app.get("/transaction/{transaction_id}", summary="Step 4 — Check transaction status")
async def get_transaction_status(transaction_id: str):
    """
    Poll this endpoint every 5 seconds after creating a transaction.
    Stop polling when status is one of: APPROVED, DECLINED, VOIDED, ERROR.

    POSSIBLE STATUSES:
      PENDING  → still processing, keep polling
      APPROVED → payment successful ✅
      DECLINED → payment failed (insufficient funds, invalid card, etc.)
      VOIDED   → transaction was reversed (card only)
      ERROR    → external error occurred
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{WOMPI_BASE_URL}/transactions/{transaction_id}",
            headers={"Authorization": f"Bearer {WOMPI_PUBLIC_KEY}"},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.json())

        tx = resp.json()["data"]
        return {
            "transaction_id": tx["id"],
            "reference": tx["reference"],
            "status": tx["status"],
            "status_message": tx.get("status_message"),
            "amount_cop": tx["amount_in_cents"] / 100,
            "payment_method_type": tx["payment_method_type"],
            "created_at": tx["created_at"],
            "is_final": tx["status"] in ("APPROVED", "DECLINED", "VOIDED", "ERROR"),
        }


# ─────────────────────────────────────────────────────────────
# BONUS: Auto-polling helper (server-side long poll)
# ─────────────────────────────────────────────────────────────
@app.get(
    "/wait-for-result/{transaction_id}",
    summary="Bonus — Server-side polling until final status",
)
async def wait_for_result(transaction_id: str, max_attempts: int = 12):
    """
    Convenience endpoint that polls internally (up to 60 seconds by default)
    and returns the final status. Use for quick testing — in production,
    implement polling on the client side or use Wompi webhooks.
    """
    async with httpx.AsyncClient() as client:
        for attempt in range(max_attempts):
            resp = await client.get(
                f"{WOMPI_BASE_URL}/transactions/{transaction_id}",
                headers={"Authorization": f"Bearer {WOMPI_PUBLIC_KEY}"},
            )
            tx = resp.json()["data"]
            if tx["status"] in ("APPROVED", "DECLINED", "VOIDED", "ERROR"):
                return {
                    "final_status": tx["status"],
                    "status_message": tx.get("status_message"),
                    "transaction_id": tx["id"],
                    "attempts": attempt + 1,
                }
            await asyncio.sleep(5)  # wait 5 seconds before retrying

    return {
        "final_status": "TIMEOUT",
        "message": "Transaction still pending after max wait time",
    }


# ─────────────────────────────────────────────────────────────
# WEBHOOK: Receive Wompi event notifications (optional but recommended)
# ─────────────────────────────────────────────────────────────
@app.post("/webhook/wompi", summary="Webhook — Receive Wompi payment events")
async def wompi_webhook(request: Request):
    """
    Configure this URL in your Wompi dashboard under "Eventos".
    Wompi sends a POST here when a transaction changes status.
    Always verify the X-Event-Checksum header to authenticate the request.
    """
    body = await request.json()
    event_type = body.get("event")
    transaction = body.get("data", {}).get("transaction", {})

    # TODO: Verify checksum header for security (see Wompi Events docs)
    # checksum = request.headers.get("X-Event-Checksum")

    print(
        f"[WEBHOOK] Event: {event_type} | TX: {transaction.get('id')} | Status: {transaction.get('status')}"
    )

    # Handle the event
    if event_type == "transaction.updated":
        status = transaction.get("status")
        reference = transaction.get("reference")
        if status == "APPROVED":
            # TODO: fulfill the order in your DB
            print(f"✅ Order {reference} paid successfully!")
        elif status in ("DECLINED", "ERROR"):
            print(f"❌ Order {reference} failed: {transaction.get('status_message')}")

    return {"received": True}


# ─────────────────────────────────────────────────────────────
# DEMO PAGE — shows the full flow visually
# ─────────────────────────────────────────────────────────────
@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Wompi Demo — $1 USD Payment</title>
      <style>
        body { font-family: Arial, sans-serif; max-width: 680px; margin: 40px auto; padding: 0 20px; background: #f7f8fa; }
        h1 { color: #1a1a2e; } h2 { color: #444; border-bottom: 2px solid #ddd; padding-bottom: 6px; }
        .step { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
        .badge { background: #5c6ac4; color: white; border-radius: 50%; width: 28px; height: 28px; display:inline-flex; align-items:center; justify-content:center; font-weight:bold; margin-right:8px; }
        input, select { width: 100%; padding: 8px; margin: 6px 0 12px; border: 1px solid #ccc; border-radius: 6px; box-sizing:border-box; }
        button { background: #5c6ac4; color: white; padding: 10px 24px; border: none; border-radius: 6px; cursor: pointer; font-size:15px; }
        button:hover { background: #3d4db7; }
        pre { background: #1e1e2e; color: #cdd6f4; padding: 14px; border-radius: 8px; overflow:auto; font-size:13px; }
        .success { color: #2e7d32; font-weight:bold; } .error { color: #c62828; font-weight:bold; }
        .note { background: #fff8e1; border-left: 4px solid #f9a825; padding: 10px 14px; border-radius:4px; font-size:13px; }
      </style>
    </head>
    <body>
      <h1>🇨🇴 Wompi Payment Demo</h1>
      <p>Charge ~<strong>$1 USD</strong> (= $4,200 COP) using the Wompi API step by step.</p>
      <div class="note">⚠️ Use <strong>Sandbox credentials</strong>. Test card: <code>4242 4242 4242 4242</code></div>

      <!-- STEP 1 -->
      <div class="step">
        <h2><span class="badge">1</span>Get Acceptance Tokens</h2>
        <p>Fetch tokens and show policy links to the user. Must be done <strong>before every payment</strong>.</p>
        <button onclick="getTokens()">Fetch Tokens</button>
        <pre id="tokens-result">—</pre>
      </div>

      <!-- STEP 2 -->
      <div class="step">
        <h2><span class="badge">2</span>Tokenize Card</h2>
        <label>Card Number</label><input id="cn" value="4242424242424242">
        <label>CVC</label><input id="cvc" value="123" style="width:80px">
        <label>Exp Month</label><input id="em" value="08" style="width:80px">
        <label>Exp Year</label><input id="ey" value="28" style="width:80px">
        <label>Cardholder Name</label><input id="ch" value="Juan Perez">
        <button onclick="tokenizeCard()">Tokenize Card</button>
        <pre id="card-result">—</pre>
      </div>

      <!-- STEP 3 -->
      <div class="step">
        <h2><span class="badge">3</span>Pay ~$1 USD</h2>
        <label>Email</label><input id="email" value="test@example.com">
        <button onclick="pay()">Pay $4,200 COP (~$1 USD)</button>
        <pre id="pay-result">—</pre>
      </div>

      <!-- STEP 4 -->
      <div class="step">
        <h2><span class="badge">4</span>Check Result</h2>
        <button onclick="checkStatus()">Poll Status</button>
        <pre id="status-result">—</pre>
      </div>

      <script>
        let stored = { acceptance_token: '', accept_personal_auth: '', card_token: '', transaction_id: '' };

        async function getTokens() {
          const r = await fetch('/acceptance-tokens'); const d = await r.json();
          stored.acceptance_token = d.acceptance_token.token;
          stored.accept_personal_auth = d.personal_data_token.token;
          document.getElementById('tokens-result').textContent = JSON.stringify(d, null, 2);
        }
        async function tokenizeCard() {
          const r = await fetch('/tokenize-card', { method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ number: document.getElementById('cn').value, cvc: document.getElementById('cvc').value,
              exp_month: document.getElementById('em').value, exp_year: document.getElementById('ey').value,
              card_holder: document.getElementById('ch').value }) });
          const d = await r.json(); stored.card_token = d.card_token;
          document.getElementById('card-result').textContent = JSON.stringify(d, null, 2);
        }
        async function pay() {
          const r = await fetch('/pay', { method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ customer_email: document.getElementById('email').value,
              card_token: stored.card_token, installments: 1,
              acceptance_token: stored.acceptance_token, accept_personal_auth: stored.accept_personal_auth }) });
          const d = await r.json(); stored.transaction_id = d.transaction_id;
          document.getElementById('pay-result').textContent = JSON.stringify(d, null, 2);
        }
        async function checkStatus() {
          if (!stored.transaction_id) { alert('Create a payment first (Step 3)'); return; }
          const r = await fetch('/wait-for-result/' + stored.transaction_id);
          const d = await r.json();
          const el = document.getElementById('status-result');
          el.textContent = JSON.stringify(d, null, 2);
          el.className = d.final_status === 'APPROVED' ? 'success' : 'error';
        }
      </script>
    </body>
    </html>
    """
