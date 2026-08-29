# Wompi + FastAPI — Complete Tutorial: Charge ~$1 USD

## What this does
A complete Python FastAPI backend to charge **~$1 USD** (≈ $4,200 COP) using **Wompi**, Colombia's payment gateway by Bancolombia.

---

## 🔑 Where to get your keys

1. Register at **https://comercios.wompi.co**
2. Go to **Developers → API Keys**
3. Copy your **Sandbox** keys (start with `pub_test_` / `prv_test_`)
4. Under **"Secretos de integración"** copy your **Integrity Secret** (`test_integrity_...`)

---

## 📦 Installation

```bash
cd wompi_tutorial
pip install -r requirements.txt
cp .env.example .env   # Then fill in your real keys
uvicorn main:app --reload
```

Then open:
- **http://localhost:8000/demo** — Interactive demo page
- **http://localhost:8000/docs** — Swagger UI with all endpoints

---

## 🔄 The Full Payment Flow

```
1. GET  /acceptance-tokens       → Fetch Wompi T&C and privacy tokens (fresh before each payment)
2. POST /tokenize-card           → Convert raw card data to a safe single-use token
3. POST /pay                     → Create the transaction (amount: $4,200 COP = ~$1 USD)
4. GET  /transaction/{id}        → Poll every 5s until status is final
```

---

## ⏰ WHEN TO FETCH THE ACCEPTANCE TOKENS (critical!)

| Moment | What to do |
|--------|-----------|
| User opens payment form | Call `GET /acceptance-tokens` |
| Show the policy links to user | Use the `permalink` URLs from the response |
| User checks both checkboxes | Store both tokens in your session |
| User submits payment | Send both tokens to `POST /pay` |
| 30+ minutes passed | Fetch new tokens — they expire! |

**Never cache tokens between sessions.** They are tied to the moment the user accepted the policy.

---

## 💰 Amount in Wompi (cents!)

| Real amount | Wompi `amount_in_cents` |
|-------------|------------------------|
| $100 COP    | 10000                  |
| $1,000 COP  | 100000                 |
| $4,200 COP (~$1 USD) | **420000** ← this tutorial |
| $10,000 COP | 1000000                |

---

## 🧪 Sandbox Test Cards

| Card Number | Result |
|-------------|--------|
| `4242 4242 4242 4242` | ✅ APPROVED |
| `4111 1111 1111 1111` | ❌ DECLINED |

Use any future expiration date and any 3-digit CVC.

---

## 📡 Transaction Statuses

| Status | Meaning |
|--------|---------|
| `PENDING` | Still processing — keep polling |
| `APPROVED` | ✅ Payment successful |
| `DECLINED` | ❌ Declined (insufficient funds, etc.) |
| `VOIDED` | Transaction reversed |
| `ERROR` | External error |

---

## 🔐 Integrity Signature

Generated server-side to prevent tampering:

```python
import hashlib
raw = f"{reference}{amount_in_cents}{currency}{INTEGRITY_SECRET}"
signature = hashlib.sha256(raw.encode("utf-8")).hexdigest()
```

**Never expose `WOMPI_INTEGRITY_SECRET` to the client.**

---

## 🌐 API Endpoints Summary

| Method | Path | Description |
|--------|------|-------------|
| GET | `/acceptance-tokens` | Fetch fresh T&C and privacy tokens |
| POST | `/tokenize-card` | Tokenize a credit/debit card |
| POST | `/pay` | Create a ~$1 USD transaction |
| GET | `/transaction/{id}` | Check transaction status |
| GET | `/wait-for-result/{id}` | Server-side polling helper |
| POST | `/webhook/wompi` | Receive Wompi event notifications |
| GET | `/demo` | Interactive demo UI |

---

## 🔔 Webhooks (production recommended)

Configure `https://yoursite.com/webhook/wompi` in your Wompi dashboard under **Eventos**.
Wompi will POST to it when transactions change status — no polling needed in production.