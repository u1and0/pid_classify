#!/usr/bin/env python3
""" End point
GET /description/ABC

フロントエンドからカテゴリ文字列ABC(英数字3桁)を受け取る
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
    class_symbol: str
    type_symbol: str
    type_string: str
    name_symbol: str
    name_string: str
    unit: str

    def __str__(self):
        """
        >>> print(categories["AAA"])
        (AA) 図番 > (A) 製品名記入
        >>> str(categories["AAA"])
        (AA) 図番 > (A) 製品名記入
        """
        x, y, z = self.class_symbol, self.type_symbol, self.name_symbol
        return f"({x}{y}) {self.type_string} > ({z}) {self.name_string}"


class Categories(UserDict):
    """
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
        self.data = {r[1] + r[2] + r[4]: Category(*r[1:]) for r in args}


def load_data(path: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    df.rename({  # 列名変更
        "分類\n記号": "class_symbol",
        "種類\n記号": "type_symbol",
        "種　類": "type_string",
        "品名\n記号": "name_symbol",
        "品　名": "name_string",
        "使用\n単位": "unit",
    }, axis=1, inplace=True)
    df.fillna(method="ffill", inplace=True)  # NAN を前の行と同じにする
    # [1:] でインデックス行を省略
    return df


df = load_data("./data/標準部品記号一覧.csv", encoding="cp932")
categories = Categories(*df.itertuples())
