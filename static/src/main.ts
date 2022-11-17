type Item = {
  name: string;
  model: string;
};
// 最上位のURL
const root: URL = new URL(window.location.href);
// index.htmlの要素
const resultDiv = document.getElementById("result");
const exampleTable = document.getElementById("example-table");

/* 予測品番の表示 */

async function postData(url: string, data: Item) {
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  return resp.json();
}

// iが0,1,2,3,4 のサイクリック
// badge色を返す
function badgeSelector(i: number): string {
  const colors = [
    "bg-primary",
    "bg-danger",
    "bg-warning",
    "bg-success",
    "bg-secondary",
  ];
  // iが4以下ならそのインデックスのcolorを返す
  if (i < colors.length) {
    return colors[i];
  }
  // iが4以上なら、再帰的にbadgeSelectorに入って
  // iが0,1,2,3,4のどれかになるまで続ける。
  return badgeSelector(i - colors.length);
}

// JSON responseを解決したら、品番カテゴリと予測確率をバッジとして表示する
function showCategoryBadges(pidMap: Map<string, number>) {
  console.debug(pidMap);
  if (resultDiv === null) return;
  resultDiv.innerHTML = ""; // Reset result div
  const h4 = document.createElement("h4");
  h4.innerHTML = "AIが予測する品番カテゴリは次のいずれかです。";
  resultDiv.appendChild(h4);
  Object.keys(pidMap).forEach((pid: string, i: number) => {
    const badge = document.createElement("button");
    if (badge === null) return;
    const proba = pidMap[pid].toPrecision(4) * 100; // 予測確率6桁 99.9999%
    badge.setAttribute("type", "button");
    badge.setAttribute("title", `予測確率${proba}%`);
    badge.classList.add("badge", "rounded-pill", badgeSelector(i)); // Bootstrap Badge
    badge.innerHTML = pid; // PID カテゴリ
    // クリックすると類似品番を表示するjsを配置
    badge.setAttribute("onclick", "getItem(this.textContent)");
    resultDiv.appendChild(badge);
  });
}

async function checkRegistered(data: Item) {
  let url = root.origin + "/search?";
  if (data.name !== "") {
    url += `name=${data.name}`;
  }
  if (data.model !== "") {
    url += `&model=${data.model}`;
  }
  await fetch(url)
    .then((resp: Response) => {
      return resp.json();
    })
    .then((item: Map<string, Item>) => {
      // GET /search で品名型式検索
      console.log(item);
      return item;
    })
    .catch(async () => {
      // 品名、型式の登録がなければ
      // POST /predict で品番予測
      const url = root.origin + "/predict";
      await postData(url, data)
        .then(showCategoryBadges)
        .catch((e: Error) => {
          console.error(e);
        });
    });
}

// 入力欄に打った情報をJSONでポスト
async function postItem() {
  const nameInput: HTMLInputElement = document.getElementById("name");
  const modelInput: HTMLInputElement = document.getElementById("model");
  const data = {
    "name": nameInput.value,
    "model": modelInput.value,
  };
  checkRegistered(data);
}

/* 類似品番テーブルの表示 */

// テーブルヘッダーの作成
function createHeader(
  table: HTMLTableElement,
  header: string[],
  caption: string,
): HTMLTableSectionElement {
  // Write table caption
  const captionElem = table.createCaption();
  captionElem.textContent = caption;
  // Write table header
  const theadElem = table.createTHead();
  const tr = theadElem.insertRow();
  header.forEach((cell: string) => {
    const th = document.createElement("th"); // th要素の追加
    th.appendChild(document.createTextNode(cell)); // thにテキスト追加
    tr.appendChild(th); // thをtrへ追加
  });
  table.appendChild(theadElem);
}

// ボタンクリックでカテゴリ検索をかけて類似品番を表示する
async function getItem(pidClass: string) {
  const url = root.origin + "/category/" + pidClass;
  const json = await fetch(url)
    .then((resp: Promise<Record<string, Item>>) => {
      return resp.json();
    })
    .catch((resp: Promise<Record<string, Item>>) => {
      return new Error(`error: ${resp.status}`);
    });
  const items: Map<string, Item> = new Map(Object.entries(json));
  console.debug(items);
  if (exampleTable === null) return;
  exampleTable.innerHTML = ""; // Reset table
  // Write table header
  createHeader(
    exampleTable, // table element
    ["品番", "品名", "型式"], // header
    `${pidClass}カテゴリに属する品名、型式をランダムに10件まで表示します。`,
  ); // caption
  const tbody = document.createElement("tbody");
  items.forEach((v: Item, k: string) => {
    const tr = tbody.insertRow();
    let td = tr.insertCell();
    td.appendChild(document.createTextNode(k));
    td = tr.insertCell();
    td.appendChild(document.createTextNode(v.name));
    td = tr.insertCell();
    td.appendChild(document.createTextNode(v.model));
  });
  exampleTable.appendChild(tbody);
}
