from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(title="Prism API")

# Without this, your browser-based frontend (running from a file:// or a
# different port than the API) would be blocked from calling this server
# at all — browsers enforce this by default as a security measure against
# malicious sites silently calling APIs on a user's behalf. "*" here means
# "allow any origin" — fine for local development; you'd tighten this to
# a specific domain if this were ever deployed publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}