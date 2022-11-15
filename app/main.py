#!/usr/bin/env python3
"""
pid_classifyのwebインターフェース
ユーザー入力の品名、型式に対して予測される品番をJSONとして返す。

Setup env
$ conda install -c uvicorn fastapi jinja2
"""
from typing import Union
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


class Item(BaseModel):
    name: str
    model: Union[str, None] = None


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
    })


@app.get("/hello")
async def hello():
    return {"message": "Hello world"}


@app.post("/item")
async def create(item: Item):
    """
    $ curl -H "Content-Type: application/json" \
    -d '{"name":"AAA", "model":"annonimous"}' \
    localhost:8880/item

    {"name":"AAA","model":"annonimous"}
    """
    print(f"received: {item}")
    json_data = [item.name + str(i) for i in range(5)]
    print(f"transfer: {json_data}")
    return json_data


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8880)
