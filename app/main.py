from contextlib import asynccontextmanager

from fastapi import FastAPI
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: runs before the first request
    yield
    # shutdown: runs when the app stops
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
