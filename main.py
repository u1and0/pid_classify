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
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pid_classify import classifier, master
from pid_category import categories

VERSION = "v0.2.5"
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class Item(BaseModel):
    """品名(name)と型式(model)の組
    { "name":"パッキン", "model":"174-452024-001"}
    """
    name: Optional[str] = None
    model: Optional[str] = None

    def like_search(self) -> pd.DataFrame:
        """グローバルオブジェクトのmaster(品番マスタ)から
        nameとmodelを含むあいまい検索"""
        dname = master["品名"].str.contains(
            self.name if self.name is not None else "")
        dmodel = master["型式"].str.contains(
            self.model if self.model is not None else "")
        return master[dname & dmodel]

    def strict_search(self) -> pd.DataFrame:
        """グローバルオブジェクトのmaster(品番マスタ)から
        nameとmodelを含む完全一致検索"""
        dname = master["品名"] == self.name
        if self.name is None:  # None なら全てTrueのSeries
            dname = ~dname
        dmodel = master["型式"] == self.model
        if self.model is None:  # None なら全てTrueのSeries
            dmodel = ~dmodel
        return master[dname & dmodel]


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
    """ 品名と型式から予測される品番カテゴリと予測確率を返す。

    ```
    $ curl -H "Content-Type: application/json" -d '{"name":"AAA", "model":"annonimous"}' localhost:8880/predict
    {"GAA":0.9482887938884507,"GJA":0.011494742934441223}
    ```
    """
    print(f"received: {item}")
    if item.name is None:
        item.name = ""
    if item.model is None:
        item.model = ""
    predict_dict = classifier.predict_proba(item.name, item.model)
    print(f"transfer: {predict_dict}")
    return predict_dict


@app.get("/pid/{parts_num}")
async def pid(parts_num: str):
    """指定品番のレコードを返す

    ```
    $ curl localhost:8880/pid/AAA-101
    {
        "カテゴリ": "AAA",
        "品名": "レドーム",
        "型式":"SW218542A"
    }
    ```
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
    カテゴリの説明を標準部品記号一覧から引く。

    ```
    $ curl 'localhost:8880/category/AAA?limit=2'
    {
      "items": {
        "AAA-1152": {
          "name": "ロープ長さ調整プレートP4 (225)",
          "model": "SZ301173-P4"
        },
        "AAA-9": {
          "name": "カバー",
          "model": "SE209548 改訂1"
        }
      },
      "text": "図番 > 製品名記入"
    }
    ```
    """
    print(f"received: {class_}")
    desc: Optional[str] = categories.get(class_)
    select = master[master["カテゴリ"] == class_]
    # これまでに登録されたことはあるが、標準部品記号一覧にはない場合
    # かつ
    # 標準部品記号一覧にあるが、これまでに登録されたことはない場合
    if (desc is None) and (len(select) < 1):
        content = {"error": f"{class_} is not exist"}
        return JSONResponse(content, status.HTTP_204_NO_CONTENT)
    if len(select) > limit:
        select = select.sample(limit)
    items = to_object(select)
    obj = {"items": items, "text": str(desc)}
    print(f"transfer: {obj}")
    return obj


@app.get("/search")
async def search(name: Optional[str] = None,
                 model: Optional[str] = None,
                 limit: int = 10,
                 strict: bool = False):
    # async def search(item: Item, limit: int = 10, strict: bool = False):
    #                  ^^^^^^^^^^
    # この書き方はGETメソッドにリクエストボディを要求してしまうので不可
    """
    クエリオプションstrictがfalsyのとき(デフォルト)
    品名、または型式、あるいはその両方から品番マスタを検索し
    クエリ文字列が含まれるレコードを返す。

    クエリオプションstrictがtruthyのとき
    品名、または型式、あるいはその両方から品番マスタを検索し
    完全一致するレコードを返す。

    品名と型式が両方指定された場合は、
    品名と型式の両方がマッチするものを返す。(AND検索)

    ```
    # マルチバイト文字はURLエンコードの必要あるので
    # このcurlリクエストはそのまま実行するとエラー
    $ curl 'localhost:8880/search?model=174-45&limit=2&strict=false'
    {
        "GFB-240": {
            "name": "パッキン",
            "model": "174-451736-002"
        },
          "GFB-169": {
            "name": "パッキン",
            "model": "174-451276-001"
        }
    }
    ```
    """
    print(f"received: name={name} model={model}")
    # ItemはNoneを受け付けるが、
    # 両方Noneのときは全てのレコードが対象になるので、
    # エラーを返しておく
    if (name is None) and (model is None):
        content = {"error": "required name or model"}
        return JSONResponse(content, status.HTTP_400_BAD_REQUEST)
    item = Item(name=name, model=model)
    # &strict=true で完全一致検索、指定なしまたはfalse であいまい検索
    select = item.strict_search() if strict else item.like_search()
    if len(select) < 1:  # 結果がなければ204 NO CONTENTエラー
        content = {"error": f"name={name} model={model} is not exist"}
        return JSONResponse(content, status.HTTP_204_NO_CONTENT)
    if len(select) > limit:  # 結果が多すぎればlimitの数だけランダムサンプリング
        select = select.sample(limit)
    obj = to_object(select)
    print(f"transfer: {obj}")
    return obj


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8880)
