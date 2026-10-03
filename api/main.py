from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.inference.generate import generate_sql
from src.inference.model import make_sql_generator


@asynccontextmanager
async def lifespan(application: FastAPI):
    adapter_dir = os.getenv("SQL_MODEL_ADAPTER_DIR")
    application.state.sql_generator = None
    if adapter_dir:
        application.state.sql_generator = make_sql_generator(
            model_name=os.getenv(
                "SQL_MODEL_NAME",
                os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
            ),
            adapter_dir=adapter_dir,
            use_4bit=os.getenv("SQL_MODEL_USE_4BIT", "true").lower() == "true",
            max_new_tokens=int(os.getenv("SQL_MODEL_MAX_NEW_TOKENS", "128")),
        )
    yield


app = FastAPI(title="SQL SLM API", version="0.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    database_schema: str = Field(default="", alias="schema", max_length=4000)
    model_response: str | None = None


class GenerateResponse(BaseModel):
    sql: str
    status: str = "ok"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    try:
        sql = generate_sql(
            request.prompt,
            request.model_response,
            getattr(http_request.app.state, "sql_generator", None),
            request.database_schema,
        )
        return GenerateResponse(sql=sql)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc
