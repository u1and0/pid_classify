#!/usr/bin/env python3
""" End point
GET /description/ABC

フロントエンドからカテゴリ文字列ABC(英字3桁)を受け取る
1~2文字目、3文字目がそれぞれ大分類、小分類にあたる。

バックエンド側ではdict[string,string]形式で(keyがカテゴリ文字列、 valueが説明文)で返す。

```
GET /description/ABC
{"ABC": (AB) 大分類の説明 >  (C) 小分類の説明}
```

"""
from collections import UserDict
from dataclasses import dataclass
import pandas as pd


@dataclass
class Category:
    """標準部品記号一覧のヘッダー行をプロパティとした
    標準部品記号一覧の各一行
    Usage:
        cat = Category('A', 'A', '図番', 'A', '製品名記入', 'EA')
    """
    class_symbol: str
    type_symbol: str
    type_string: str
    name_symbol: str
    name_string: str
    unit: str

    def __str__(self):
        """
        >>> print(categories["AAA"])
        (AAA) 図番 >  製品名記入
        >>> str(categories["AAA"])
        (AAA) 図番 >  製品名記入
        """
        x, y, z = self.class_symbol, self.type_symbol, self.name_symbol
        return f"({x}{y}{z}) {self.type_string} > {self.name_string}"


class Categories(UserDict):
    """Categoryクラスの集まりをディクショナリとして表現したクラス。
    キーにはclass_symbol, type_symbol, name_symbolの英字3桁を結合したもの、
    値にはCategoryをセットする。

    Usage:
        >>> df = load_data("./data/標準部品記号一覧.csv", encoding="cp932")
        >>> categories = Categories(*df.itertuples())

        >>> categories
        {'AAA': Category(class_symbol='A', type_symbol='A', type_string='図番',
        name_symbol='A', name_string='製品名記入',
        unit='EA・SE・組・台・箱・式'), 'ABA': Category(class_symbol='A', t
        ...

        >>> categories["AAA"]
        Category(class_symbol='A',
                type_symbol='A',
                type_string='図番',
                name_symbol='A',
                name_string='製品名記入',
                unit='EA・SE・組・台・箱・式')
    """
    def __init__(self, *args):
        # [1:] でインデックス行を省略
        self.data = {r[1] + r[2] + r[4]: Category(*r[1:]) for r in args}


def load_data(path: str, **kwargs) -> pd.DataFrame:
    """標準部品記号一覧.csvを読み込んで、データフレーム化する。
    前処理として、 印刷用の繰り返しヘッダー行を削除してから
    標準部品記号一覧.xlsを 標準部品記号一覧.csvとして保存する。
    """
    return pd.read_csv(path, **kwargs)\
        .rename({  # 列名変更
            "分類\n記号": "class_symbol",
            "種類\n記号": "type_symbol",
            "種　類": "type_string",
                "品名\n記号": "name_symbol",
                "品　名": "name_string",
                "使用\n単位": "unit",
                }, axis=1)\
        .fillna(method="ffill")  # NAN を前の行と同じにする


df = load_data("./data/標準部品記号一覧.csv", encoding="cp932")
categories = Categories(*df.itertuples())
