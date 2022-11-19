// JSONで返ってくる品名、型式のペア
type Item = {
  name: string;
  model: string;
};
type Pid = string;
type Items = Map<Pid, Item>;
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
  // GET /search で品名型式検索
  await fetch(url)
    .then((resp: Promise<Items>) => {
      console.log(resp.status, resp.statusText);
      if (resp.status === 200) { // 品名、型式が完全一致した場合
        return resp.json();
      } else { // 完全一致検索できなかった場合
        // 204ステータスなのでエラーを投げる。
        // catch先で、POST /predictして予測を返す
        throw new Error(`${resp.status}: ${resp.statusText}`);
      }
    })
    .then((items: Items) => {
      console.log("search items: ", items);
      // MapキャストしないとObjectとして渡されて、forEach使えない
      items = new Map(Object.entries(items));
      createTable(items, "タイトル");
    })
    .catch((e: Error) => {
      console.debug(e); // 品名、型式の完全一致が見つからなかった204エラー
      // POST /predict で品番予測
      const url = root.origin + "/predict";
      postData(url, data)
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
  if (data.name === "") {
    const msg = "品名を必ず入力してください。";
    console.error(msg);
    resultDiv.innerHTML = ""; // Reset result div
    const errorMessage = document.createElement("div");
    errorMessage.classList.add("alert", "alert-warning");
    errorMessage.setAttribute("role", "alert");
    errorMessage.innerHTML = msg;
    resultDiv.appendChild(errorMessage);
    return;
  }
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

function createTable(items: Map<string, Item>, caption: string) {
  if (exampleTable === null) return;
  exampleTable.innerHTML = ""; // Reset table
  // Write table header
  createHeader(
    exampleTable, // table element
    ["品番", "品名", "型式"], // header
    caption,
  ); // caption
  const tbody = document.createElement("tbody");
  console.log("search items: ", items);
  // items = new Map(Object.entries(items));
  // を差し込むと完全一致検索の方はテーブルが表示されるが、
  // カテゴリ検索のテーブルは表示されない
  items.forEach((v: Item, k: string) => {
    console.log(`key: ${k}, value: ${v}`);
    const tr = tbody.insertRow(); // 行要素の作成
    // セルを3列追加
    let td = tr.insertCell();
    td.appendChild(document.createTextNode(k));
    td = tr.insertCell();
    td.appendChild(document.createTextNode(v.name));
    td = tr.insertCell();
    td.appendChild(document.createTextNode(v.model));
    tbody.appendChild(tr); // 行を追加
  });
  exampleTable.appendChild(tbody);
}

// ボタンクリックでカテゴリ検索をかけて類似品番を表示する
async function getItem(pidClass: string) {
  const url = root.origin + "/category/" + pidClass;
  const json = await fetch(url)
    .then((resp: Promise<Items>) => {
      return resp.json();
    })
    .catch((resp: Promise<Items>) => {
      return new Error(`error: ${resp.status}: ${resp.statusText}`);
    });
  const items: Items = new Map(Object.entries(json));
  console.debug(items);
  const caption = `${pidClass}カテゴリに属する品名、型式をランダムに10件まで表示します。`;
  createTable(items, caption);
}
