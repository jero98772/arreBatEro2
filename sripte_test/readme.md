Here's a complete tutorial on how to process a $1 payment with Stripe in Python + FastAPI.

---

## Stripe $1 Payment with FastAPI — Complete Tutorial

### Overview of the flow

```
Client (React/HTML)
    │
    ├─ POST /create-payment-intent  →  FastAPI creates a PaymentIntent
    │       (returns client_secret)
    │
    ├─ Stripe.js confirms payment   →  Stripe processes card
    │
    └─ POST /webhook                →  FastAPI receives confirmed event
             (tokens issued HERE ✅)
```

---

### 1. Install dependencies

```bash
pip install fastapi uvicorn stripe python-dotenv
```

---

### 2. `.env` file

```env
STRIPE_SECRET_KEY=sk_test_YOUR_SECRET_KEY
STRIPE_WEBHOOK_SECRET=whsec_YOUR_WEBHOOK_SECRET
```

---

### 3. `main.py` — Full FastAPI backend

```python
import os
import stripe
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow frontend to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")


# ──────────────────────────────────────────────
# Step 1: Create a PaymentIntent for $1.00
# ──────────────────────────────────────────────
class CreatePaymentRequest(BaseModel):
    user_id: str  # so you know WHO is paying


@app.post("/create-payment-intent")
async def create_payment_intent(body: CreatePaymentRequest):
    try:
        intent = stripe.PaymentIntent.create(
            amount=100,           # Amount in cents → $1.00
            currency="usd",
            metadata={"user_id": body.user_id},  # store user info
            automatic_payment_methods={"enabled": True},
        )
        return {"client_secret": intent.client_secret}

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# Step 2: Webhook — fires AFTER payment succeeds
# 🪙 THIS IS WHERE YOU ISSUE TOKENS 🪙
# ──────────────────────────────────────────────
@app.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="stripe-signature"),
):
    payload = await request.body()

    # Verify the event came from Stripe (not a fake request)
    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # ✅ Payment confirmed — issue tokens NOW
    if event["type"] == "payment_intent.succeeded":
        intent = event["data"]["object"]
        user_id = intent["metadata"].get("user_id")
        amount_cents = intent["amount"]          # 100 = $1.00
        payment_id = intent["id"]

        print(f"✅ Payment succeeded!")
        print(f"   User:    {user_id}")
        print(f"   Amount:  ${amount_cents / 100:.2f}")
        print(f"   Payment: {payment_id}")

        # 🪙 Issue tokens here — examples:
        tokens_to_grant = calculate_tokens(amount_cents)
        await grant_tokens_to_user(user_id, tokens_to_grant, payment_id)

    # Payment failed — notify user if needed
    elif event["type"] == "payment_intent.payment_failed":
        intent = event["data"]["object"]
        user_id = intent["metadata"].get("user_id")
        print(f"❌ Payment failed for user {user_id}")

    return JSONResponse({"status": "ok"})


# ──────────────────────────────────────────────
# Helper: calculate tokens based on payment
# ──────────────────────────────────────────────
def calculate_tokens(amount_cents: int) -> int:
    """$1.00 = 100 tokens. Scale linearly."""
    return amount_cents  # 100 cents → 100 tokens


# ──────────────────────────────────────────────
# Helper: grant tokens (replace with your DB logic)
# ──────────────────────────────────────────────
async def grant_tokens_to_user(user_id: str, tokens: int, payment_id: str):
    """
    Replace this with your actual database update.
    Examples:
      - SQLAlchemy:  db.execute("UPDATE users SET tokens = tokens + ? WHERE id = ?", tokens, user_id)
      - MongoDB:     await users.update_one({"_id": user_id}, {"$inc": {"tokens": tokens}})
      - Redis:       await redis.incrby(f"tokens:{user_id}", tokens)
    """
    print(f"🪙  Granted {tokens} tokens to user {user_id} (payment: {payment_id})")
```

---

### 4. Frontend — `index.html`

```html
<!DOCTYPE html>
<html>
<head>
  <title>Pay $1</title>
  <script src="https://js.stripe.com/v3/"></script>
</head>
<body>
  <h2>Pay $1.00 to get tokens</h2>
  <div id="payment-element"></div>
  <button id="pay-btn">Pay $1.00</button>
  <p id="message"></p>

  <script>
    const stripe = Stripe("pk_test_YOUR_PUBLISHABLE_KEY");
    const userId = "user_123"; // from your auth system

    async function init() {
      // 1. Ask backend to create a PaymentIntent
      const res = await fetch("http://localhost:8000/create-payment-intent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId }),
      });
      const { client_secret } = await res.json();

      // 2. Mount Stripe's payment UI
      const elements = stripe.elements({ clientSecret: client_secret });
      const paymentElement = elements.create("payment");
      paymentElement.mount("#payment-element");

      // 3. Handle Pay button
      document.getElementById("pay-btn").onclick = async () => {
        const { error } = await stripe.confirmPayment({
          elements,
          confirmParams: {
            return_url: "http://localhost:8000/success", // redirect after pay
          },
        });
        if (error) {
          document.getElementById("message").textContent = error.message;
        }
      };
    }

    init();
  </script>
</body>
</html>
```

---

### 5. Run the server

```bash
uvicorn main:app --reload --port 8000
```

Then expose your webhook locally with the Stripe CLI:

```bash
stripe listen --forward-to localhost:8000/webhook
```

The CLI prints your `WEBHOOK_SECRET` — paste it into your `.env`.

---

### 🪙 When to issue tokens — the golden rule

| Event | Issue tokens? | Why |
|---|---|---|
| `/create-payment-intent` called | ❌ No | Payment not confirmed yet |
| Frontend calls `confirmPayment` | ❌ No | Can still fail or be fraudulent |
| `payment_intent.succeeded` webhook | ✅ **YES** | Stripe guarantees money is captured |
| User redirected to `return_url` | ❌ No | User can fake/skip this redirect |

**Always issue tokens inside the `payment_intent.succeeded` webhook.** That's the only event that Stripe guarantees is real and final. Never trust the frontend redirect alone.

---

### Production checklist

- Use `sk_live_` and `pk_live_` keys (not `sk_test_`)
- Store `payment_id` in your DB to prevent duplicate token grants (idempotency)
- Restrict `allow_origins` in CORS to your real domain
- Use HTTPS for your webhook endpoint
- Set up Stripe's webhook signing in the Dashboard, not just via CLI