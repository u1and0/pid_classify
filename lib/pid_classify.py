#!/usr/bin/env python3
import os
from collections import namedtuple
from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# PidMaster : 学習データ(品名、型式)、教師データ(品番カテゴリ)のデータコンテナ
PidMaster = namedtuple("PidMaster", ["names", "models", "categories"])


def load_data(datapath: str) -> PidMaster:
    """csvファイルを読み込んで品番マスタデータを返す"""
    cwd = os.path.dirname(__file__)
    df = pd.read_csv(cwd + "/" + datapath)
    pid = df["品番"].values
    names = df["品名"].values
    models = df["型式"].values
    categories = [i.split("-")[0] for i in pid]
    return PidMaster(names, models, categories)


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

    def predict_proba(self, name, model: str, top: int = 5) -> pd.Series:
        """品名と型式の組から想定される品番カテゴリの上位 top件を返す"""
        namemodel = f"{name}\t{model}"
        vec = self.vectorizer.fit_transform([namemodel])
        prob = self.clf.predict_proba(vec)
        index = self.le.inverse_transform(self.clf.classes_)
        se = pd.Series(prob[0], index).sort_values(ascending=False)
        return se[:top]

    def predict_mask_proba(self,
                           name: str,
                           model: str,
                           top=100,
                           threshold=0.95) -> list[str]:
        """ 品名と型式の組から想定される品番カテゴリを複数返す。
        品番のサジェスト確率の累計がthresholdを超えるまでのリストを返す
        """
        pid_series = self.predict_proba(name, model, top)
        se_iter = pid_series.iteritems()
        predict_list = []
        cumsum_percentile = 0
        # 確率の合計が閾値を超えるまでイテレート
        while cumsum_percentile < threshold:
            pid, prob = next(se_iter)
            cumsum_percentile += prob
            predict_list.append(pid)
        return predict_list

    def score(self):
        """テストデータによるスコア算出"""
        return self.clf.score(self.X_test, self.y_test)


def training(pid_master: PidMaster) -> PidClassify:
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
        f"{n}\t{m}" for n, m in zip(pid_master.names, pid_master.models))
    # カテゴリ文字列を数値に変換
    le = LabelEncoder()
    y = le.fit_transform(pid_master.categories)
    # データをトレーニング用、テスト用に分割します。
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # ナイーブベイズにより学習させます。
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return PidClassify(clf, vectorizer, le, X_train, X_test, y_train, y_test)


### main ###
master = load_data("../data/pidmaster.csv")
classifier = training(master)
# 学習の評価
score = classifier.score()
assert 0.60 < score < 0.99, "適切な精度で学習できていません。"
print(f"学習精度 {score:.4}で学習を完了しました。")

# Usage
#
# se = clf.predict_proba(vectorizer, le, name="ブレーカ\tBBW351", top=100)
# print(se)
# predict_noproba(se)
#
#
# 同じインターフェースで、上位5件を順位付きで表示しました。
# 確率が分散しているので、上位5件を超えたところで閾値を超えためです。
#
# 順位すらいらない、不要な情報だ、ということであれば、listをsetにすればいいだけです。
#
#
# predict_set = set(predict_noproba(se))
# print(f"品名: ブレーカ 型式:BBW351 の品番は95%以上の確率で{predict_set}のいずれかです。")
