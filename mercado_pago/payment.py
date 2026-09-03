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
                "currency_id": currency,  # e.g. "COP", "USD", "ARS"
                "unit_price": amount,  # e.g. 1.0 for $1
            }
        ],
        "back_urls": {
            "success": back_urls.get(
                "success", "http://localhost:8000/payment/success"
            ),
            "failure": back_urls.get(
                "failure", "http://localhost:8000/payment/failure"
            ),
            "pending": back_urls.get(
                "pending", "http://localhost:8000/payment/pending"
            ),
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
