// type Item = {
//   "品名": string,
//   "型式": string,
// }
// type Row = Record<string,Item>
type Query = {
  name: string;
  model: string;
};
type Label = Map<string, number>;
const root: URL = new URL(window.location.href);

async function postData(url: string, data: Query) {
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

// 入力欄に打った情報をJSONでポスト
function postItem() {
  const url = root.origin + "/predict";
  const nameInput: HTMLInputElement = document.getElementById("name");
  if (nameInput === null) return;
  const modelInput: HTMLInputElement = document.getElementById("model");
  if (modelInput === null) return;
  const data = {
    "name": nameInput.value,
    "model": modelInput.value,
  };
  postData(url, data)
    .then((pidMap: Label) => {
      console.log(pidMap); // DEBUG
      const resultDiv = document.getElementById("result");
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
        resultDiv.appendChild(badge);
      });
    })
    .catch((e: Error) => {
      console.error(e);
    });
}
