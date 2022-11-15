#!/usr/bin/env python3
import os
from collections import namedtuple
from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

PidMaster = namedtuple("PidMaster", ["names", "models", "categories"])


def load_data(datapath: str) -> PidMaster:
    cwd = os.path.dirname(__file__)
    df = pd.read_csv(cwd + "/" + datapath)
    pid = df["品番"].values
    name = df["品名"].values
    types = df["型式"].values
    pid_category = [i.split("-")[0] for i in pid]
    return PidMaster(name, types, pid_category)


@dataclass
class PidClassify:
    clf: MultinomialNB
    vectorizer: HashingVectorizer
    le: LabelEncoder

    def predict_pid(self, clf: MultinomialNB, vectorizer: HashingVectorizer,
                    le: LabelEncoder, name: str) -> str:
        """
        適当な品名 name を入れて、
        学習機により提案された品番カテゴリを一つ返す
        学習済みの学習機clfを引数に取る。
        """
        vec = self.vectorizer.fit_transform([name])
        category_idx = self.clf.predict(vec)
        predict_pid = self.le.inverse_transform(category_idx)
        return predict_pid[0]

    # MultinominalNBクラスのメソッドとしてpredict_pidを登録
    # clf.predict_pid(name)として実行できる
    # setattr(MultinomialNB, "predict_pid", predict_pid)

    def prob_series(self,
                    clf: MultinomialNB,
                    vectorizer: HashingVectorizer,
                    le: LabelEncoder,
                    name: str,
                    top: int = 5) -> pd.Series:
        """適当な品名nameが属する品番カテゴリの上位 top件を返す"""
        vec = self.vectorizer.fit_transform([name])
        prob = self.clf.predict_proba(vec)
        index = self.le.inverse_transform(self.clf.classes_)
        se = pd.Series(prob[0], index).sort_values(ascending=False)
        return se[:top]

    # MultinominalNBクラスのメソッドとしてpredict_pidを登録
    # clf.predict_pid(name)として実行できる
    # setattr(MultinomialNB, "prob_series", prob_series)

    @staticmethod
    def pid_mask_probability(pid_series: pd.Series,
                             threshold=0.95) -> list[str]:
        """品番のサジェスト確率の累計がthresholdを超えるまでのリストを返す"""
        se_iter = pid_series.iteritems()
        predict_list = []
        cumsum_percentile = 0
        # 確率の合計が閾値を超えるまでイテレート
        while cumsum_percentile < threshold:
            pid, prob = next(se_iter)
            cumsum_percentile += prob
            predict_list.append(pid)
        return predict_list


def training(pid_master: PidMaster):
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

    # 学習の評価
    score = clf.score(X_test, y_test)
    assert 0.60 < score < 0.99, "適切な精度で学習できていません。"
    print(f"学習精度 {score:.4}で学習を完了しました。")
    return PidClassify(clf, vectorizer, le)


if __name__ == "__main__":
    master = load_data("../data/pidmaster.csv")
    classifier = training(master)

    # Usage
    #
    # se = clf.prob_series(vectorizer, le, name="ブレーカ\tBBW351", top=100)
    # print(se)
    # pid_mask_probability(se)
    #
    #
    # 同じインターフェースで、上位5件を順位付きで表示しました。
    # 確率が分散しているので、上位5件を超えたところで閾値を超えためです。
    #
    # 順位すらいらない、不要な情報だ、ということであれば、listをsetにすればいいだけです。
    #
    #
    # predict_set = set(pid_mask_probability(se))
    # print(f"品名: ブレーカ 型式:BBW351 の品番は95%以上の確率で{predict_set}のいずれかです。")
