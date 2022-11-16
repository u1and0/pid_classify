#!/usr/bin/env python3
"""
pid_classifyのwebインターフェース
ユーザー入力の品名、型式に対して予測される品番をJSONとして返す。

Setup env
$ conda install -c uvicorn fastapi jinja2
"""
from typing import Union
import uvicorn
from fastapi import FastAPI, Request, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pid_classify import classifier, master


class Item(BaseModel):
    name: str
    model: Union[str, None] = None


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root(request: Request):
    """/indexへリダイレクト"""
    return RedirectResponse("/index")


@app.get("/index")
async def index(request: Request):
    """メインページ"""
    return templates.TemplateResponse("index.html", {
        "request": request,
    })


@app.get("/hello")
async def hello():
    """サーバー生きているか確認"""
    return {"message": "品番予測AI heart beat"}


@app.post("/predict")
async def predict(item: Item):
    """ classifierへJSONポストし、品番カテゴリ予測をJSONで受け取る
    $ curl -H "Content-Type: application/json" \
        -d '{"name":"AAA", "model":"annonimous"}' \
        localhost:8880/item
    ["AAA", "ZBA"]
    """
    print(f"received: {item}")
    json_list = classifier.predict_mask_proba(item.name, item.model)
    print(f"transfer: {json_list}")
    proba = classifier.predict_proba(item.name, item.model)
    print(proba)  # 確率表示
    return json_list


@app.get("/pid/{pid}")
async def pid(pid: str):
    """指定品番のレコードを返す
    $ curl localhost:8880/pid/AAA-1001
    {
        AAA-1001: {
            品名: "シリンダ",
            型式: "QB764"
        }
    }
    """
    try:
        return master.loc[pid].to_dict()
    except KeyError:
        content = {"error": f"{pid} is not exist"}
        return JSONResponse(content, status.HTTP_404_NOT_FOUND)


@app.get("/category/{class_}")
async def category(class_: str):
    """指定カテゴリに属するレコードを返す
    $ curl localhost:8880/category/AAA
    {
        AAA-1: {
            品名: "シリンダ",
            型式: "A745"
        },
        AAA-2: {
            品名: "シリンダ",
            型式: "B153"
        }
    }
    """
    obj = master[master["カテゴリ"] == class_].T.to_dict()
    if len(obj) < 1:
        content = {"error": f"{class_} is not exist"}
        return JSONResponse(content, status.HTTP_404_NOT_FOUND)
    return obj


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8880)
