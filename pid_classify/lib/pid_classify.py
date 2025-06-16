#!/usr/bin/env python3
"""品番カテゴリ分類ライブラリ
Usage

品番カテゴリ分類器クラスをインポートしてファクトリメソッドで初期化します。

>>> from pid_classify.lib.pid_classify import Classifier
>>> classifier = Classifier.create_and_train('/path/to/db.db')
品番データを学習中...
学習精度 0.7618で学習を完了しました。
>>> n = "ブレーカ", "BBW351"

品名と型式の組から、学習器により提案された品番カテゴリを一つ返します。
>>> classifier.predict(*n)
'SCA'

品名と型式の組から想定される品番カテゴリ確率の上位を返します。
>>> classifier._predict_proba_series(*n).head(5)
SCA    0.905975
KCD    0.065212
ABA    0.024902
GAA    0.002606
AZB    0.000705
dtype: float64

品名と型式の組から想定される品番カテゴリを複数返します。
デフォルトでは累計0.95になるまでの確率のディクショナリを返します。

>>> classifier.predict_proba(*n)
{'SCA': 0.9059745176959081, 'KCD': 0.06521197399595013}

品名と型式の組から想定される品番カテゴリを複数返します。
デフォルトでは累計0.95になるまでの確率の上位順リストを返します。
>>> classifier.predict_mask_proba(*n)
['SCA', 'KCD']
"""

import os
from datetime import datetime
import hashlib
import sqlite3
import pandas as pd
import logging
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import NotFittedError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_data(filepath: str, query: str, **kwargs) -> pd.DataFrame:
    """sqlite3 DBファイルを読み込んで品番マスタデータを返す"""
    try:
        logger.info(f"Loading data from {filepath}")
        con = sqlite3.connect(filepath)

        df = pd.read_sql(query, con, index_col="品番", **kwargs)
    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {e}")
        raise
    finally:
        con.close()

    if df.empty:
        raise ValueError("Loaded DataFrame is empty")

    # Handle missing values
    if "品名" in df.columns:
        df["品名"] = df["品名"].fillna("")
    if "型式" in df.columns:
        df["型式"] = df["型式"].fillna("")

    # 行頭のAAAなどのアルファベット文字列のみ抽出
    df["カテゴリ"] = [str(i).split("-")[0] for i in df.index]

    logger.info(f"Successfully loaded {len(df)} records")
    return df


def training(
    pid_master: pd.DataFrame,
) -> tuple[MultinomialNB, TfidfVectorizer, LabelEncoder, float, dict]:
    """品番データを学習して学習器を生成する
    下記データフレームの品名と型式のタブ文字区切りを説明変数に
    品番を目的変数に置き換えて学習する。

    pd.DataFrame ==
             品名     型式
    品番
    AAA-123  ケーブル type123
    ...
    """
    logger.info("Starting model training")

    # テキストをTF-IDF特徴量に変換
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 3),  # trigram
        max_features=50000,  # Feature数を制限
        min_df=2,  # 最低2回出現する特徴量のみ使用
        max_df=0.95,  # 95%以上のドキュメントに出現する特徴量は除外
    )

    # 品名 / 型式をタブ区切り
    X = vectorizer.fit_transform(
        f"{n}\t{m}" for n, m in zip(pid_master["品名"], pid_master["型式"])
    )

    # カテゴリ文字列を数値に変換
    le = LabelEncoder()
    y = le.fit_transform(pid_master["カテゴリ"].values)

    # カテゴリごとのサンプル数をチェック
    unique, counts = np.unique(y, return_counts=True)
    min_samples = np.min(counts)

    if min_samples < 2:
        logger.warning(
            f"Some categories have only {min_samples} sample(s). Removing stratification."
        )
        # stratifyを使わずに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
    else:
        # データをトレーニング用、テスト用に分割します。
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0, stratify=y
        )

    # ナイーブベイズにより学習させます。
    clf = MultinomialNB(alpha=0.1)  # スムージングパラメータを調整
    clf.fit(X_train, y_train)

    # より詳細な評価
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    # 分類レポートと混同行列を生成
    # テストセットに含まれるクラスのみを対象にする
    unique_test_classes = np.unique(y_test)
    target_names = le.inverse_transform(unique_test_classes)

    report = classification_report(
        y_test,
        y_pred,
        labels=unique_test_classes,
        target_names=target_names,
        output_dict=True,
    )
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_test_classes)

    evaluation_metrics = {
        "accuracy": score,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "target_names": target_names,
    }

    logger.info(f"Training completed with accuracy: {score:.4f}")

    return clf, vectorizer, le, score, evaluation_metrics


class Classifier:
    """品番カテゴリ分類器クラス"""

    def __init__(self, data: pd.DataFrame):
        """
        learn from SQLite3 filepath

        Properties:
            clf: MultinomialNB
            vectorizer: TfidfVectorizer
            le: LabelEncoder
            score: float
            evaluation_metrics: dict
        Methods:
            predict: str
            _predict_proba_series: pd.Series
            predict_proba: dict[str:float]
            predict_mask_proba: list[str]
            save_model: None
        """
        try:
            self.clf, self.vectorizer, self.le, self.score, self.evaluation_metrics = (
                training(data)
            )
            logger.info(f"学習精度 {self.score:.4}で学習を完了しました。")
            assert 0.60 < self.score < 1, "適切な精度で学習できていません。"
        except Exception as e:
            logger.error(f"Classifier initialization failed: {e}")
            raise

    @staticmethod
    def create_and_train(master: pd.DataFrame):
        """分類器を作成・学習して返すファクトリメソッド"""
        try:
            logger.info("品番データを学習中...")
            return Classifier(master)
        except Exception as e:
            logger.error(f"分類器の作成に失敗しました: {e}")
            raise

    def save_model(self, filepath: str):
        """訓練済みモデルをファイルに保存"""
        try:
            model_data = {
                "classifier": self.clf,
                "vectorizer": self.vectorizer,
                "label_encoder": self.le,
                "score": self.score,
                "evaluation_metrics": self.evaluation_metrics,
            }
            with open(filepath, "wb") as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

    @staticmethod
    def load_model(filepath: str):
        """保存されたモデルを読み込み"""
        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            # 新しいClassifierインスタンスを作成（データなしで初期化）
            classifier = object.__new__(Classifier)
            classifier.clf = model_data["classifier"]
            classifier.vectorizer = model_data["vectorizer"]
            classifier.le = model_data["label_encoder"]
            classifier.score = model_data["score"]
            classifier.evaluation_metrics = model_data["evaluation_metrics"]

            logger.info(f"Model loaded from {filepath}")
            return classifier
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, name, model: str) -> str:
        """品名と型式の組から、
        学習器により提案された品番カテゴリを一つ返す
        """
        try:
            namemodel = f"{name}\t{model}"
            vec = self.vectorizer.transform(
                [namemodel]
            )  # fit_transformではなくtransform
            category_idx = self.clf.predict(vec)
            predict_pid = self.le.inverse_transform(category_idx)
            return predict_pid[0]
        except NotFittedError:
            logger.error("Model not fitted. Please train the model first.")
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def _predict_proba_series(self, name, model: str) -> pd.Series:
        """品名と型式の組から想定される
        品番カテゴリ確率を返す
        """
        try:
            namemodel = f"{name}\t{model}"
            vec = self.vectorizer.transform(
                [namemodel]
            )  # fit_transformではなくtransform
            prob = self.clf.predict_proba(vec)
            index = self.le.inverse_transform(self.clf.classes_)
            se = pd.Series(prob[0], index).sort_values(ascending=False)
            return se
        except NotFittedError:
            logger.error("Model not fitted. Please train the model first.")
            raise
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise

    def predict_proba(
        self, name: str, model: str, top=100, threshold=0.95
    ) -> dict[str, float]:
        """品名と型式の組から想定される品番カテゴリを複数返す。
        デフォルトでは累計thresholdになるまでの
        確率の上位順ディクショナリを返す。
        """
        try:
            pid_series = self._predict_proba_series(name, model).head(top)
            se_iter = pid_series.items()
            predict_dict = {}
            cumsum_percentile = 0
            # 確率の合計が閾値を超えるまでイテレート
            for pid, prob in se_iter:
                if cumsum_percentile > threshold:
                    break
                cumsum_percentile += prob
                predict_dict[pid] = prob
            return predict_dict
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise

    def predict_mask_proba(
        self, name: str, model: str, top=100, threshold=0.95
    ) -> list[str]:
        """品名と型式の組から想定される品番カテゴリを複数返す。
        デフォルトでは累計thresholdになるまでの
        確率の上位順リストを返す。
        """
        try:
            dic = self.predict_proba(name, model, top, threshold)
            return list(dic.keys())
        except Exception as e:
            logger.error(f"Probability mask prediction failed: {e}")
            raise

    def get_evaluation_metrics(self) -> dict:
        """評価メトリクスを返す"""
        return self.evaluation_metrics


class Master(pd.DataFrame):
    """pd.DataFrameを基底クラスとした
    データ、ファイルプロパティホルダー
    Data and file property holder based on pd.DataFrame
        self:pd.DataFrame = training data
        date:str = file create time
        hash:str = file content hash
    """

    def __init__(self, filepath: str, **kwargs):
        """データをfilepathから読み込み、
        ファイルのctimeとhashをプロパティへ設定する。
        """
        try:
            query = "SELECT 品番, 品名, 型式 FROM 品番"
            _data: pd.DataFrame = _load_data(filepath, query, **kwargs)
            super().__init__(_data)
            _ctime: float = os.path.getctime(filepath)
            self.date = datetime.fromtimestamp(_ctime)
            with open(filepath, "rb") as f:
                _b = f.read()
            self.hash: str = hashlib.sha256(_b).hexdigest()
            logger.info(f"Master data initialized with {len(_data)} records")
        except Exception as e:
            logger.error(f"Failed to initialize Master: {e}")
            raise


class MiscMaster(pd.DataFrame):
    """
    諸口品マスタ
    部品手配テーブルからS_から始まる品番とその品名を重複なしに取得した
    """

    def __init__(self, filepath: str, **kwargs):
        query = "SELECT DISTINCT 品番,品名 FROM 部品手配 WHERE 品番 LIKE 'S_%'"
        _data = _load_data(filepath, query=query, **kwargs)

        _data["型式"] = ""
        # _data["カテゴリ"] = _data["品番"]

        super().__init__(_data)

        # Masterクラスと同じプロパティ
        # 更新日を取得
        _ctime: float = os.path.getctime(filepath)
        self.date = datetime.fromtimestamp(_ctime)
        # データハッシュを取得
        with open(filepath, "rb") as f:
            _b = f.read()
        self.hash = hashlib.sha256(_b).hexdigest()
        logger.info(f"MiscMaster initialized with {len(_data)} records")


class MiscClassifier:
    """諸口品の品名から品番を予測する分類器"""

    def __init__(self, data: pd.DataFrame):
        self._classifier = Classifier(data)

    def predict(self, name: str) -> str:
        """品名から予測される品番を一つ返す"""
        return self._classifier.predict(name, "")

    def predict_proba(self, name: str, **kwargs) -> dict[str, float]:
        """品名から予測される品番と確率を返す"""
        return self._classifier.predict_proba(name, "", **kwargs)

    def predict_mask_proba(self, name: str, **kwargs) -> list[str]:
        """品名から予測される品番のリストを返す"""
        dic = self._classifier.predict_proba(name, "", **kwargs)
        return list(dic.keys())
