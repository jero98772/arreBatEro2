from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import mercadopago
import os
from dotenv import load_dotenv
from payment import create_preference

load_dotenv()

app = FastAPI(title="MercadoPago Payment API")

APP_ENV = os.getenv("APP_ENV", "sandbox")


# ─────────────────────────────────────────
# POST /payment/create
# Creates a $1 payment preference and
# returns the checkout URL
# ─────────────────────────────────────────
@app.post("/payment/create")
async def create_payment():
    try:
        preference = create_preference(
            title="Test Payment",
            amount=1.0,  # $1 USD
            currency="USD",  # Change to "COP" if charging in pesos
            back_urls={
                "success": "http://localhost:8000/payment/success",
                "failure": "http://localhost:8000/payment/failure",
                "pending": "http://localhost:8000/payment/pending",
            },
            # notification_url="https://your-public-url.com/payment/webhook",
        )

        # In sandbox, use sandbox_init_point
        # In production, use init_point
        if APP_ENV == "sandbox":
            checkout_url = preference["sandbox_init_point"]
        else:
            checkout_url = preference["init_point"]

        return {
            "preference_id": preference["id"],
            "checkout_url": checkout_url,
            "env": APP_ENV,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# GET /payment/success
# MercadoPago redirects here after payment
# ─────────────────────────────────────────
@app.get("/payment/success")
async def payment_success(
    collection_id: str = None,
    collection_status: str = None,
    payment_id: str = None,
    status: str = None,
    external_reference: str = None,
    payment_type: str = None,
    merchant_order_id: str = None,
    preference_id: str = None,
    site_id: str = None,
    processing_mode: str = None,
    merchant_account_id: str = None,
):
    # MercadoPago sends payment details as query params
    return {
        "message": "Payment approved! ✅",
        "payment_id": payment_id,
        "status": status,
        "preference_id": preference_id,
    }


# ─────────────────────────────────────────
# GET /payment/failure
# ─────────────────────────────────────────
@app.get("/payment/failure")
async def payment_failure():
    return {"message": "Payment failed ❌"}


# ─────────────────────────────────────────
# GET /payment/pending
# ─────────────────────────────────────────
@app.get("/payment/pending")
async def payment_pending():
    return {"message": "Payment pending ⏳"}


# ─────────────────────────────────────────
# POST /payment/webhook
# Receives real-time notifications from MP
# ─────────────────────────────────────────
@app.post("/payment/webhook")
async def payment_webhook(request: Request):
    body = await request.json()
    print("Webhook received:", body)

    # MercadoPago sends a "type" field to tell you what happened
    if body.get("type") == "payment":
        payment_id = body["data"]["id"]

        # You can query the payment here:
        sdk = mercadopago.SDK(os.getenv("MERCADOPAGO_ACCESS_TOKEN"))
        result = sdk.payment().get(payment_id)
        payment_info = result["response"]

        print(f"Payment {payment_id} status: {payment_info['status']}")
        # "approved", "rejected", "pending", etc.

    return JSONResponse(content={"received": True}, status_code=200)
