from fastapi import FastAPI


app = FastAPI(
    title="Offline Local Retrieval Service",
    version="0.1.0",
)


@app.get("/health/live")
def liveness() -> dict[str, str]:
    return {
        "status": "ok",
    }

