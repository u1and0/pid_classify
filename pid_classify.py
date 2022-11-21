#!/usr/bin/env python3
"""品番カテゴリ分類ライブラリ
Usage

品番カテゴリ分類器クラスをインポートします。
インポートした時点で学習を完了しています。
>>> from lib.pid_classify import classifier
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
from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def load_data(datapath: str) -> pd.DataFrame:
    """csvファイルを読み込んで品番マスタデータを返す"""
    cwd = os.path.dirname(__file__)
    df = pd.read_csv(
        cwd + "/" + datapath,
        index_col=0,
        usecols=[0, 1, 2],
        skiprows=1,
        skipfooter=1,
        encoding="cp932",
        engine="python",  # required by skipfooter option
    )
    df["カテゴリ"] = [i.split("-")[0] for i in df.index]
    return df.loc[:, ["カテゴリ", "品名", "型式"]]


@dataclass
class PidClassify:
    """品番カテゴリ分類器クラス"""
    clf: MultinomialNB
    vectorizer: HashingVectorizer
    le: LabelEncoder
    X_train: list
    X_test: list
    y_train: list
    y_test: list

    def predict(self, name, model: str) -> str:
        """ 品名と型式の組から、
        学習器により提案された品番カテゴリを一つ返す
        """
        namemodel = f"{name}\t{model}"
        vec = self.vectorizer.fit_transform([namemodel])
        category_idx = self.clf.predict(vec)
        predict_pid = self.le.inverse_transform(category_idx)
        return predict_pid[0]

    def _predict_proba_series(self, name, model: str) -> pd.Series:
        """ 品名と型式の組から想定される
        品番カテゴリ確率を返す
        """
        namemodel = f"{name}\t{model}"
        vec = self.vectorizer.fit_transform([namemodel])
        prob = self.clf.predict_proba(vec)
        index = self.le.inverse_transform(self.clf.classes_)
        se = pd.Series(prob[0], index).sort_values(ascending=False)
        return se

    def predict_proba(self,
                      name: str,
                      model: str,
                      top=100,
                      threshold=0.95) -> dict[str:float]:
        """ 品名と型式の組から想定される品番カテゴリを複数返す。
        デフォルトでは累計thresholdになるまでの
        確率の上位順ディクショナリを返す。
        """
        pid_series = self._predict_proba_series(name, model).head(top)
        se_iter = pid_series.iteritems()
        predict_dict = {}
        cumsum_percentile = 0
        # 確率の合計が閾値を超えるまでイテレート
        for pid, prob in se_iter:
            if cumsum_percentile > threshold:
                break
            cumsum_percentile += prob
            predict_dict[pid] = prob
        return predict_dict

    def predict_mask_proba(self,
                           name: str,
                           model: str,
                           top=100,
                           threshold=0.95) -> list[str]:
        """ 品名と型式の組から想定される品番カテゴリを複数返す。
        デフォルトでは累計thresholdになるまでの
        確率の上位順リストを返す。
        """
        dic = self.predict_proba(name, model, top, threshold)
        return list(dic.keys())

    def score(self):
        """テストデータによるスコア算出"""
        return self.clf.score(self.X_test, self.y_test)


def training(pid_master: pd.DataFrame) -> PidClassify:
    """品番データを学習して学習器を生成する"""
    # テキストをtrigram特徴量に変換
    vectorizer = HashingVectorizer(
        n_features=2**16,
        analyzer="char",
        ngram_range=(3, 3),  # trigram
        binary=True,
        norm=None)
    # 品名 / 型式をタブ区切り
    X = vectorizer.fit_transform(
        f"{n}\t{m}" for n, m in zip(pid_master["品名"], pid_master["型式"]))
    # カテゴリ文字列を数値に変換
    le = LabelEncoder()
    y = le.fit_transform(pid_master["カテゴリ"].values)
    # データをトレーニング用、テスト用に分割します。
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # ナイーブベイズにより学習させます。
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return PidClassify(clf, vectorizer, le, X_train, X_test, y_train, y_test)


### main ###
print("品番データを学習中...")
master: pd.DataFrame = load_data("data/pidmaster.csv")
classifier = training(master)
# 学習の評価
score = classifier.score()
print(f"学習精度 {score:.4}で学習を完了しました。")
assert 0.60 < score < 1, "適切な精度で学習できていません。"
