Here's a complete tutorial for integrating Mercado Pago payments with FastAPI in Python.

---

## 🛒 Mercado Pago + FastAPI — $1 Payment Tutorial

### Overview of the Flow

```
User → FastAPI → MercadoPago API → Checkout URL → User pays → Webhook → FastAPI confirms
```

---

### 🔑 When to Get Your Tokens

**Go to MercadoPago Developers right now, before writing any code:**

1. Go to [https://www.mercadopago.com.co/developers/panel](https://www.mercadopago.com.co/developers/panel) *(Colombia, since you're in Cali)*
2. Create or open an app
3. You'll find **two sets of tokens**:

| Token | When to use |
|---|---|
| **Test Access Token** | Development & testing (sandbox) |
| **Production Access Token** | Real payments (live) |

> ⚠️ **Start with the Test token.** Never commit your Production token to git.

---

### 📁 Project Structure

```
mercadopago_api/
├── main.py
├── payment.py
├── .env
└── requirements.txt
```

---

### 1. Install Dependencies

```bash
pip install fastapi uvicorn mercadopago python-dotenv httpx
```

`requirements.txt`:
```txt
fastapi
uvicorn
mercadopago
python-dotenv
httpx
```

---

### 2. `.env` — Store Your Token Here

```env
# Use TEST token during development
MERCADOPAGO_ACCESS_TOKEN=TEST-XXXXXXXXXXXXXXXXXXXX-XXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-XXXXXXXX

# Set to "sandbox" or "production"
APP_ENV=sandbox
```

> 🔐 **Where to find the token:**
> - Log in at mercadopago.com.co
> - Go to **Developers → My Apps → Your App → Credentials**
> - Copy the **Test Access Token** (starts with `TEST-`)
> - Only switch to the **Production Access Token** when you're ready to go live

---

### 3. `payment.py` — Payment Logic

```python
import mercadopago
import os
from dotenv import load_dotenv

load_dotenv()

def get_sdk():
    access_token = os.getenv("MERCADOPAGO_ACCESS_TOKEN")
    if not access_token:
        raise ValueError("MERCADOPAGO_ACCESS_TOKEN not set in .env")
    return mercadopago.SDK(access_token)


def create_preference(
    title: str,
    amount: float,
    currency: str,
    back_urls: dict,
    notification_url: str | None = None,
):
    sdk = get_sdk()

    preference_data = {
        "items": [
            {
                "title": title,
                "quantity": 1,
                "currency_id": currency,   # e.g. "COP", "USD", "ARS"
                "unit_price": amount,      # e.g. 1.0 for $1
            }
        ],
        "back_urls": {
            "success": back_urls.get("success", "http://localhost:8000/payment/success"),
            "failure": back_urls.get("failure", "http://localhost:8000/payment/failure"),
            "pending": back_urls.get("pending", "http://localhost:8000/payment/pending"),
        },
        "auto_return": "approved",  # Redirect automatically on success
    }

    # Optional: webhook to receive real-time notifications
    if notification_url:
        preference_data["notification_url"] = notification_url

    result = sdk.preference().create(preference_data)

    if result["status"] != 201:
        raise Exception(f"MercadoPago error: {result['response']}")

    return result["response"]
```

---

### 4. `main.py` — FastAPI App

```python
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
            amount=1.0,          # $1 USD
            currency="USD",      # Change to "COP" if charging in pesos
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
```

---

### 5. Run the Server

```bash
uvicorn main:app --reload --port 8000
```

Then hit:

```bash
curl -X POST http://localhost:8000/payment/create
```

You'll get back a `checkout_url` — open it in the browser, and you'll see the MercadoPago sandbox checkout for $1.

---

### 🧪 Testing with Sandbox Cards

MercadoPago provides test cards. For Colombia:

| Card | Number | CVV | Expiry |
|---|---|---|---|
| Visa (approved) | `4013 5406 8274 6260` | `123` | Any future date |
| Mastercard (approved) | `5031 7557 3453 0604` | `123` | Any future date |
| Any (rejected) | `4000 0000 0000 0002` | `123` | Any future date |

Use any test email and any name.

---

### 🚀 Going to Production — Token Checklist

| Step | Action |
|---|---|
| 1 | Finish all tests with `TEST-` token |
| 2 | Go to **Developers → Credentials → Production** |
| 3 | Copy the **Production Access Token** (starts with `APP_USR-`) |
| 4 | Set `MERCADOPAGO_ACCESS_TOKEN=APP_USR-...` in your production `.env` |
| 5 | Set `APP_ENV=production` |
| 6 | Use `init_point` instead of `sandbox_init_point` (the code already handles this) |
| 7 | Set a real public `notification_url` for webhooks (use ngrok locally to test) |

---

### 💡 Currency Note

Since you're in Cali, Colombia, note that MercadoPago Colombia uses **COP (Colombian Pesos)**. If you want to charge exactly $1 USD worth, you'd need to convert. If you just want to charge a symbolic amount in COP:

```python
amount=1000.0,   # COP $1,000 ≈ ~$0.25 USD
currency="COP",
```

To charge in USD, keep `currency="USD"` and `amount=1.0` — MercadoPago supports it for some integrations.