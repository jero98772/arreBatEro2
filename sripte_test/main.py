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
            amount=100,  # Amount in cents → $1.00
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
        amount_cents = intent["amount"]  # 100 = $1.00
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
