import os
import uuid
from datetime import datetime, timedelta, timezone

import anyio
import jwt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel, Field

load_dotenv()

GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
JWT_SECRET = os.environ.get("JWT_SECRET", "change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24  # 1 day

app = FastAPI(title="FastAPI + Google + JWT Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fake "database" (use a real DB in production)
users_db: dict[str, dict] = {}
items_db: dict[str, list[dict]] = {}  # sub -> items


# ---------- Schemas ----------
class GoogleTokenIn(BaseModel):
    id_token: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class ProfileUpdate(BaseModel):
    name: str | None = None
    bio: str | None = Field(default=None, max_length=500)


class ItemIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    content: str = Field(default="", max_length=2000)


# ---------- Helpers ----------
def create_jwt(user: dict) -> str:
    payload = {
        "sub": user["sub"],
        "email": user["email"],
        "name": user.get("name"),
        "picture": user.get("picture"),
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(authorization: str = Header(default="")) -> dict:
    """Dependency: extracts and validates the JWT from the Authorization header."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")


def users_items(user: dict) -> list[dict]:
    return items_db.setdefault(user["sub"], [])


# ---------- Auth routes ----------
@app.post("/auth/google", response_model=TokenOut)
async def auth_google(body: GoogleTokenIn):
    # verify_oauth2_token is blocking → run it in a thread, keep the loop free
    try:
        info = await anyio.to_thread.run_sync(
            lambda: google_id_token.verify_oauth2_token(
                body.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
            )
        )
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Google token")

    user = users_db.get(info["sub"]) or {
        "sub": info["sub"],
        "email": info["email"],
        "name": info.get("name"),
        "picture": info.get("picture"),
        "bio": "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    users_db[user["sub"]] = user

    return TokenOut(access_token=create_jwt(user), user=user)


# ---------- 🔒 Members-only sections ----------
@app.get("/me")
async def read_me(user: dict = Depends(get_current_user)):
    """Who am I + full profile."""
    return users_db.get(user["sub"], user)


@app.get("/dashboard")
async def dashboard(user: dict = Depends(get_current_user)):
    """Personal dashboard stats."""
    items = users_items(user)
    return {
        "message": f"Welcome back, {user.get('name', 'user')}!",
        "email": user["email"],
        "picture": user.get("picture"),
        "member_since": users_db.get(user["sub"], {}).get("created_at"),
        "stats": {
            "item_count": len(items),
            "items_completed": sum(1 for i in items if i.get("done")),
        },
    }


@app.put("/profile")
async def update_profile(body: ProfileUpdate, user: dict = Depends(get_current_user)):
    """Edit own profile (name, bio)."""
    record = users_db.setdefault(user["sub"], dict(user))
    if body.name is not None:
        record["name"] = body.name
    if body.bio is not None:
        record["bio"] = body.bio
    return record


@app.get("/items")
async def list_items(user: dict = Depends(get_current_user)):
    """List MY items only — each user only sees their own."""
    return users_items(user)


@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item(body: ItemIn, user: dict = Depends(get_current_user)):
    item = {"id": uuid.uuid4().hex[:8], **body.model_dump(), "done": False}
    users_items(user).append(item)
    return item


@app.patch("/items/{item_id}/toggle")
async def toggle_item(item_id: str, user: dict = Depends(get_current_user)):
    for item in users_items(user):
        if item["id"] == item_id:
            item["done"] = not item["done"]
            return item
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")


@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: str, user: dict = Depends(get_current_user)):
    items = users_items(user)
    for i, item in enumerate(items):
        if item["id"] == item_id:
            items.pop(i)
            return
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")
