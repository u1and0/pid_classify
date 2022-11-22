#!/usr/bin/env python3
"""
pid_classifyのwebインターフェース
ユーザー入力の品名、型式に対して予測される品番をJSONとして返す。

Setup env
$ conda install -c uvicorn fastapi jinja2
"""
from typing import Optional
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, status
from pydantic import BaseModel
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pid_classify import classifier, master


class Item(BaseModel):
    """フロントエンドで定義したItem型のJSONデータ形式
    { "name":"パッキン", "model":"174-452024-001"}
    """
    name: Optional[str] = None
    model: Optional[str] = None


VERSION = "v0.2.2"
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def to_object(df: pd.DataFrame) -> dict[str, Item]:
    """ DataFrameをItem型へ変換"""
    renamer = {"品名": "name", "型式": "model"}
    renamed = df.rename(renamer, axis=1).loc[:, ["name", "model"]]
    return renamed.T.to_dict()


@app.get("/")
async def root():
    """/indexへリダイレクト"""
    return RedirectResponse("/index")


@app.get("/index")
async def index(request: Request):
    """メインページ"""
    return templates.TemplateResponse(
        "index.html", {
            "request": request,
            "version": VERSION,
            "score": classifier.score,
            "date": master.date,
            "hash": master.hash,
            "registered": len(master),
            "categories": len(set(master["カテゴリ"])),
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
        localhost:8880/predict
    ["AAA", "ZBA"]
    """
    print(f"received: {item}")
    predict_dict = classifier.predict_proba(item.name, item.model)
    print(f"transfer: {predict_dict}")
    return predict_dict


@app.get("/pid/{parts_num}")
async def pid(parts_num: str):
    """指定品番のレコードを返す
    $ curl localhost:8880/pid/AAA-1001
    {
        AAA-1001: {
            品名: "シリンダ",
            型式: "QB764"
        }
    }
    """
    print(f"received: {parts_num}")
    try:
        obj = master.loc[parts_num].to_dict()
        print(f"transfer: {obj}")
        return obj
    except KeyError:
        content = {"error": f"{parts_num} is not exist"}
        return JSONResponse(content, status.HTTP_204_NO_CONTENT)


@app.get("/category/{class_}")
async def category(class_: str, limit: int = 10):
    """指定カテゴリに属するレコードをランダムにlimit件返す
    $ curl localhost:8880/category/AAA&limit=2
    {
        AAA-1: {
            name: "シリンダ",
            model: "A745"
        },
        AAA-2: {
            name: "シリンダ",
            model: "B153"
        }
    }
    """
    print(f"received: {class_}")
    select = master[master["カテゴリ"] == class_]
    if len(select) < 1:
        content = {"error": f"{class_} is not exist"}
        return JSONResponse(content, status.HTTP_204_NO_CONTENT)
    if len(select) > limit:
        select = select.sample(limit)
    obj = to_object(select)
    print(f"transfer: {obj}")
    return obj


@app.get("/search")
async def search(name: str, model: Optional[str] = None, limit: int = 10):
    """品名、または型式、あるいはその両方から品番マスタを検索し
    JSONとしてレコードを返す。
    # マルチバイト文字はURLエンコードの必要あるので
    # このcurlリクエストはそのまま実行するとエラー
    $ curl localhost:8880/search?name=パッキン&model=174-452024-001
    {
        "GFB-9":{
            "name":"パッキン",
            "model":"174-452024-001"
            }
    }
    """
    print(f"received: name={name} model={model}")
    dname = master["品名"] == name
    if name is None:  # None なら全てTrueのSeries
        dname = ~dname
    dmodel = master["型式"] == model
    if model is None:  # None なら全てTrueのSeries
        dmodel = ~dmodel
    select = master[dname & dmodel]
    if len(select) < 1:
        content = {"error": f"name={name} model={model} is not exist"}
        return JSONResponse(content, status.HTTP_204_NO_CONTENT)
    if len(select) > limit:
        select = select.sample(limit)
    obj = to_object(select)
    print(f"transfer: {obj}")
    return obj


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8880)
