#!/usr/bin/env python3
"""
pid_classifyのwebインターフェース
ユーザー入力の品名、型式に対して予測される品番をJSONとして返す。

Setup env
$ conda install -c uvicorn fastapi jinja2
"""

import os
from typing import Optional
import logging
from threading import Lock
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, status, Response
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pid_classify.lib.pid_category import Categories
from pid_classify.lib.pid_classify import (
    Master,
    MiscMaster,
    Classifier,
    MiscClassifier,
    DataLoader,
)

VERSION = "v0.3.0"

# 再学習間隔の設定（時間単位）
RETRAIN_INTERVAL_HOURS = int(os.environ.get("RETRAIN_INTERVAL_HOURS", "1"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """モデルの状態を管理するクラス"""

    def __init__(self):
        self.master = None
        self.classifier = None
        self.misc_master = None
        self.misc_classifier = None
        self.categories = None
        self.last_training_time = None
        self.training_lock = Lock()

    def is_training(self) -> bool:
        """現在学習中かどうかを返す"""
        return self.training_lock.locked()

    def is_ready(self) -> bool:
        """モデルが利用可能かどうかを返す"""
        return (
            self.master is not None
            and not self.master.empty
            and self.classifier is not None
            and self.misc_master is not None
            and not self.misc_master.empty
            and self.misc_classifier is not None
            and self.categories is not None
        )

    def get_training_status(self) -> dict:
        """学習状態の情報を返す"""
        return {
            "is_training": self.is_training(),
            "last_training_time": self.last_training_time,
            "has_models": all(
                [
                    self.master is not None,
                    self.classifier is not None,
                    self.misc_master is not None,
                    self.misc_classifier is not None,
                    self.categories is not None,
                ]
            ),
        }

    def get_status(self) -> dict:
        """モデルの状態情報を返す"""
        return {
            "is_training": self.is_training(),
            "last_training_time": self.last_training_time,
            "model_loaded": self.classifier is not None,
            "misc_model_loaded": self.misc_classifier is not None,
            "registered_count": len(self.master) if self.master is not None else 0,
            "categories_count": len(set(self.master["カテゴリ"]))
            if self.master is not None
            else 0,
            "ready": self.is_ready(),
        }


# グローバルモデル管理インスタンス
model_manager = ModelManager()

# スケジューラーの初期化
scheduler = AsyncIOScheduler()


async def train_models():
    """モデルの学習を行う関数"""
    if model_manager.training_lock.locked():
        logger.warning("Training already in progress, skipping")
        return

    with model_manager.training_lock:
        try:
            logger.info("Starting model training...")

            # Initialize classifier and master data
            db_path = os.path.join(os.path.dirname(__file__), "data", "cwz.db")
            metadata = DataLoader.create_file_metadata(db_path)

            # 品番マスタの作成
            query = "SELECT 品番, 品名, 型式 FROM 品番"
            pid_df: pd.DataFrame = DataLoader.load(db_path, query)
            master = Master(pid_df, metadata)
            classifier = Classifier.create_and_train(master)

            # 部品手配テーブルから諸口品マスタの作成
            query = "SELECT DISTINCT 品番,品名 FROM 部品手配 WHERE 品番 LIKE 'S_%'"
            misc_df: pd.DataFrame = DataLoader.load(db_path, query)
            misc_master = MiscMaster(misc_df, metadata)
            misc_classifier = MiscClassifier.create_and_train(misc_master)

            # カテゴリ登録
            csv_path = os.path.join(
                os.path.dirname(__file__), "data", "標準部品記号一覧.csv"
            )
            df = Categories.load_data(csv_path, encoding="cp932")
            categories = Categories(*df.itertuples())

            # モデルを更新
            model_manager.master = master
            model_manager.classifier = classifier
            model_manager.misc_master = misc_master
            model_manager.misc_classifier = misc_classifier
            model_manager.categories = categories
            model_manager.last_training_time = pd.Timestamp.now()

            logger.info(
                f"Model training completed successfully. Score: {classifier.score}"
            )

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーション起動時の処理"""
    logger.info("Starting PID Classify application...")

    # 初回モデル学習
    await train_models()

    # 設定された間隔での再学習スケジュールを設定
    scheduler.add_job(
        train_models,
        trigger=IntervalTrigger(hours=RETRAIN_INTERVAL_HOURS),
        id="periodic_retrain",
        name=f"Periodic Model Retraining ({RETRAIN_INTERVAL_HOURS}h interval)",
        replace_existing=True,
    )

    # スケジューラー開始
    scheduler.start()
    logger.info(
        f"Scheduler started - models will retrain every {RETRAIN_INTERVAL_HOURS} hour(s)"
    )

    yield

    # Shutdown 処理
    logger.info("Shutting down PID Classify application...")

    # スケジューラー停止
    scheduler.shutdown()
    logger.info("Scheduler stopped")


# サーバー設定
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class CategoryPredictRequest(BaseModel):
    """品番カテゴリ予測用のリクエストモデル"""

    name: str
    model: str = ""
    threshold: float = 0.95
    top: int = 100


class MiscPredictRequest(BaseModel):
    """諸口品番予測用のリクエストモデル"""

    name: str
    threshold: float = 0.95
    top: int = 100


class Item(BaseModel):
    """品名(name)と型式(model)の組
    { "name":"パッキン", "model":"174-452024-001"}
    """

    name: Optional[str] = None
    model: Optional[str] = None

    def like_search(self) -> pd.DataFrame:
        """グローバルオブジェクトのmaster(品番マスタ)から
        nameとmodelを含むあいまい検索"""
        master = model_manager.master
        if master is None:
            return pd.DataFrame()
        dname = master["品名"].str.contains(self.name if self.name is not None else "")
        dmodel = master["型式"].str.contains(
            self.model if self.model is not None else ""
        )
        return master[dname & dmodel]

    def strict_search(self) -> pd.DataFrame:
        """グローバルオブジェクトのmaster(品番マスタ)から
        nameとmodelを含む完全一致検索"""
        master = model_manager.master
        if master is None:
            return pd.DataFrame()
        dname = master["品名"] == self.name
        if self.name is None:  # None なら全てTrueのSeries
            dname = ~dname
        dmodel = master["型式"] == self.model
        if self.model is None:  # None なら全てTrueのSeries
            dmodel = ~dmodel
        return master[dname & dmodel]


def to_object(df: pd.DataFrame) -> dict[str, Item]:
    """DataFrameをItem型へ変換"""
    renamer = {"品名": "name", "型式": "model"}
    renamed = df.rename(renamer, axis=1).loc[:, ["name", "model"]]
    return renamed.T.to_dict()


def add_deprecation_headers(
    response: Response, new_endpoint: str, sunset_date: str = "2026-12-31"
):
    """廃止予定ヘッダーの追加"""
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = sunset_date
    response.headers["Link"] = f"<{new_endpoint}>; rel='successor-version'"
    response.headers["Warning"] = (
        f"299 - 'This endpoint is deprecated and will be removed on {sunset_date}'"
    )


@app.get("/")
async def root():
    """/indexへリダイレクト"""
    return RedirectResponse("/index")


@app.get("/index")
async def index(request: Request):
    """メインページ"""
    if model_manager.is_ready():
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "version": VERSION,
                "score": model_manager.classifier.score,
                "date": model_manager.master.date,
                "hash": model_manager.master.hash,
                "registered": len(model_manager.master),
                "categories": len(set(model_manager.master["カテゴリ"])),
            },
        )
    else:
        return templates.TemplateResponse(
            "loading.html",
            {
                "request": request,
                "version": VERSION,
                "message": "Models are loading. Please wait...",
            },
        )


@app.get("/hello")
async def hello():
    """サーバー生きているか確認"""
    return {"message": "品番予測AI heart beat"}


@app.post("/predict/category")
async def predict_category(request: CategoryPredictRequest):
    """品名と型式から予測される品番カテゴリと予測確率を返す。
    Args:
        request: 品名、型式、しきい値、上位件数を含むリクエスト

    Returns:
        品番カテゴリと予測確率の辞書

    Example:
    ```
            curl -H "Content-Type: application/json" \
             -d '{"name":"ブレーカ", "model":"BBW351", "threshold":0.95}' \
             localhost:8880/predict/category
    ```
    """
    logger.info(
        f"Category prediction request: name={request.name}, model={request.model}"
    )

    if model_manager.classifier is None:
        content = {
            "error": "Model not available. Please wait for training to complete."
        }
        return JSONResponse(content, status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        predict_dict = model_manager.classifier.predict_proba(
            request.name,
            request.model,
            threshold=request.threshold,
            top=request.top,
        )
        logger.info(f"Category prediction result: {predict_dict}")
        return predict_dict
    except Exception as e:
        logger.error(f"Category prediction failed: {e}")
        content = {"error": f"Prediction failed: {str(e)}"}
        return JSONResponse(content, status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/predict/misc")
async def predict_misc(request: MiscPredictRequest):
    """品名から予測される諸口品番カテゴリと予測確率を返す。

    Args:
        request: 品名、しきい値、上位件数を含むリクエスト

    Returns:
        諸口品番と予測確率の辞書

    Example:
        ```
        $ curl -H "Content-Type: application/json" \
            -d '{"name":"シリコンゴム", "threshold":0.95}' \
            localhost:8880/predict/misc
        {"S_HOZAI":0.8287681977055886,"S_SHOMO":0.11253231047825198,"S_ZAIRYO":0.04253258924038836}
        ```
    """
    logger.info(f"Misc prediction request: name={request.name}")

    if model_manager.misc_classifier is None:
        content = {
            "error": "Misc model not available. Please wait for training to complete."
        }
        return JSONResponse(content, status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        predict_dict = model_manager.misc_classifier.predict_proba(
            request.name,
            threshold=request.threshold,
            top=request.top,
        )
        logger.info(f"Misc prediction result: {predict_dict}")
        return predict_dict
    except Exception as e:
        logger.error(f"Misc prediction failed: {e}")
        content = {"error": f"Prediction failed: {str(e)}"}
        return JSONResponse(content, status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.post("/predict")
async def predict(item: Item, response: Response):
    """【廃止予定】品名と型式から予測される品番カテゴリと予測確率を返す

    ⚠️ このエンドポイントは廃止予定です。
    新しいエンドポイント /predict/category をご利用ください。
    """
    # 廃止予定ヘッダーを追加
    add_deprecation_headers(response, "/predict/category")

    logger.warning(
        f"Deprecated endpoint /predict accessed. Redirecting to /predict/category"
    )

    # 旧形式から新形式へ変換
    name = item.name or ""
    model = item.model or ""

    logger.info(f"Deprecated predict request: name={name}, model={model}")

    if model_manager.classifier is None:
        content = {
            "error": "Model not available. Please wait for training to complete."
        }
        return JSONResponse(content, status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        predict_dict = model_manager.classifier.predict_proba(name, model)
        logger.info(f"Deprecated predict result: {predict_dict}")
        return predict_dict
    except Exception as e:
        logger.error(f"Deprecated predict failed: {e}")
        content = {"error": f"Prediction failed: {str(e)}"}
        return JSONResponse(content, status.HTTP_500_INTERNAL_SERVER_ERROR)


# API情報エンドポイント
@app.get("/api/info")
async def api_info():
    """API情報とエンドポイント一覧を返す"""
    return {
        "version": VERSION,
        "endpoints": {
            "current": {
                "POST /predict/category": "品名と型式から品番カテゴリを予測",
                "POST /predict/misc": "品名から諸口品番を予測",
            },
            "deprecated": {
                "POST /predict": "【廃止予定: 2025-12-31】 /predict/category を使用してください",
            },
        },
        "migration_guide": {
            "old_predict": {
                "endpoint": "POST /predict",
                "new_endpoint": "POST /predict/category",
                "example_old": '{"name": "ブレーカ", "model": "BBW351"}',
                "example_new": '{"name": "ブレーカ", "model": "BBW351", "threshold": 0.95}',
            },
        },
    }


@app.get("/api/training-status")
async def training_status():
    """学習状態とスケジューラーの情報を返す"""
    status = model_manager.get_training_status()
    status.update(
        {
            "retrain_interval_hours": RETRAIN_INTERVAL_HOURS,
            "scheduler_running": scheduler.running if scheduler else False,
            "next_retrain_time": None,
        }
    )

    # 次の再学習時間の取得
    if scheduler and scheduler.running:
        job = scheduler.get_job("periodic_retrain")
        if job and job.next_run_time:
            status["next_retrain_time"] = job.next_run_time.isoformat()

    return status


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

    if model_manager.master is None:
        content = {
            "error": "Master data not available. Please wait for training to complete."
        }
        return JSONResponse(content, status.HTTP_503_SERVICE_UNAVAILABLE)

    try:
        obj = model_manager.master.loc[parts_num].to_dict()
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

    if model_manager.categories is None or model_manager.master is None:
        content = {"error": "Data not available. Please wait for training to complete."}
        return JSONResponse(content, status.HTTP_503_SERVICE_UNAVAILABLE)

    desc: Optional[str] = model_manager.categories.get(class_)
    select = model_manager.master[model_manager.master["カテゴリ"] == class_]
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
async def search(
    name: Optional[str] = None,
    model: Optional[str] = None,
    limit: int = 10,
    strict: bool = False,
):
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
