from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import answer_with_router

app = FastAPI(title="Interview API", version="0.1.0")


class AnswerRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer_question(payload: AnswerRequest) -> AnswerResponse:
    try:
        _, answer = answer_with_router(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not answer:
        raise HTTPException(status_code=500, detail="Empty response from model.")

    return AnswerResponse(answer=answer)
