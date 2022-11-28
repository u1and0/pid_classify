// Bootstrap alerts labels
type Level =
  | "alert-primary"
  | "alert-secondary"
  | "alert-success"
  | "alert-danger"
  | "alert-warning"
  | "alert-info"
  | "alert-light"
  | "alert-dark";
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
const nameInput: HTMLInputElement = document.getElementById("name");
const modelInput: HTMLInputElement = document.getElementById("model");
const nameDataList: HTMLElement = document.getElementById("name-list");
const modelDataList: HTMLElement = document.getElementById("model-list");

// エントリポイントアクセス後の状態をメッセージで表示
function resultAlertLabel(msg: string, level: Level) {
  resultDiv.innerHTML = ""; // Reset result div
  const label = document.createElement("div");
  label.classList.add("alert", level);
  label.setAttribute("role", "alert");
  label.innerHTML = msg;
  resultDiv.appendChild(label);
}

// div の表示非表示切り替え
function toggleMenu(elem: string) {
  const obj = document.getElementById(elem).style;
  obj.display = (obj.display == "none") ? "block" : "none";
}

/* 予測品番の表示 */

// POST method helper
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
  const msg = "AIが予測する品番カテゴリは次のいずれかです。";
  resultAlertLabel(msg, "alert-success");
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

// 過去に品番登録したことがある品名、型式を完全一致検索して
// 10行まで表示する
// 品名型式が完全一致するレコードが1行もなかった場合、
// 品番予測AIによる予測カテゴリを表示する
async function checkRegistered(data: Item) {
  let url = root.origin + "/search?strict=true";
  if (data.name !== "") {
    url += `&name=${data.name}`;
  }
  if (data.model !== "") {
    url += `&model=${data.model}`;
  }
  // GET /search で品名型式検索
  await fetch(url)
    .then((resp) => {
      console.debug(resp.status, resp.statusText);
      if (resp.status === 200) { // 品名、型式が完全一致した場合
        return resp.json();
      } else { // 完全一致検索できなかった場合
        // 204ステータスなのでエラーを投げる。
        // catch先で、POST /predictして予測を返す
        throw new Error(`${resp.status}: ${resp.statusText}`);
      }
    })
    .then((items: Items) => {
      console.debug("search items: ", items);
      // ラベル表示
      resultAlertLabel("品番登録済みです。", "alert-info");
      // テーブル表示
      // MapキャストしないとObjectとして渡されて、forEach使えない
      items = new Map(Object.entries(items));
      createTable(
        items,
        `品名: ${data.name}, 型式: ${data.model} で\
          登録されている品番をランダムに10件まで表示します。`,
      );
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

// APIで取得したJSON(Items型)でテーブル作成
function createTable(items: Map<string, Item>, caption: string) {
  if (exampleTable === null) return;
  exampleTable.innerHTML = ""; // Reset table
  // Write table header
  createHeader(
    exampleTable, // table element
    ["品番", "品名", "型式"], // header
    caption,
  );
  const tbody = document.createElement("tbody");
  // console.debug("search items: ", items);
  // items = new Map(Object.entries(items));
  // を差し込むと完全一致検索の方はテーブルが表示されるが、
  // カテゴリ検索のテーブルは表示されない
  items.forEach((v: Item, k: string) => {
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

// datalistタグをoptionで埋める
// 引数にdatalistのエレメントとjsonで取得したoptionTextsが必要
function completionList(element: HTMLElement, optionTexts: Set<string>) {
  element.innerHTML = ""; // reset datalist
  console.debug(optionTexts);
  // 型式一覧を補完候補へ挿入
  optionTexts.forEach((n: string) => {
    const optionElem = document.createElement("option");
    optionElem.innerHTML = n;
    element.appendChild(optionElem);
  });
}

/* ここから下の関数は
* onclick属性でhtml上の要素に定義するので
* decleared but never read
*/

// 入力欄に打った情報をJSONでポスト
async function postItem() {
  const data = {
    "name": nameInput.value.trim(),
    "model": modelInput.value.trim(),
  };
  if (data.name === "") {
    const msg = "品名を必ず入力してください。";
    console.error(msg);
    resultAlertLabel(msg, "alert-warning");
    return;
  }
  checkRegistered(data);
}

// ボタンクリックでカテゴリ検索をかけて類似品番を表示する
// pidClass は英字3文字
async function getItem(pidClass: string) {
  const url = root.origin + "/category/" + pidClass.trim();
  await fetch(url)
    .then((resp) => {
      return resp.json();
    })
    .then((json) => {
      console.debug(json);
      const items: Items = new Map(Object.entries(json.items));
      createTable(
        items,
        `${pidClass}カテゴリの分類規則は[ ${json.text} ]です。` +
          `同一カテゴリの品名、型式をランダムに10件まで表示します。`,
      );
    })
    .catch((resp) => {
      return new Error(`error: ${resp.status}: ${resp.statusText}`);
    });
}

// 入力があるたびに品名一覧をinputの補完へ挿入
function fetchList(partsName: string) {
  nameDataList.innerHTML = ""; // reset datalist
  modelDataList.innerHTML = ""; // reset datalist
  partsName = partsName.trim();
  if (partsName === "") return;
  let timeoutID;
  clearTimeout(timeoutID); // 前回のタイマーストップ
  timeoutID = setTimeout(() => {
    let url = `${root.origin}/search?limit=30&name=${partsName}`;
    console.debug(url);
    fetch(url)
      .then((resp) => {
        return resp.json();
      })
      .then((json) => {
        // /searchの結果を重複無しで品名datalistと型式datalistへ加える
        const items: Items = new Map(Object.entries(json)); // Itemsへキャスト
        console.debug(items);
        const nameSet: Set<string> = new Set();
        const modelSet: Set<string> = new Set();
        items.forEach((item: Item) => {
          nameSet.add(item.name);
          modelSet.add(item.model);
        });
        completionList(nameDataList, nameSet);
        completionList(modelDataList, modelSet);
      })
      .catch((resp) => {
        return new Error(`error: ${resp.status}: ${resp.statusText}`);
      });
  }, 1500); // 1.5秒入力がなければ、品名一覧をfetch
}
