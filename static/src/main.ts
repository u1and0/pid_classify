const root: URL = new URL(window.location.href);

async function postData(url: string, data: Record<string, unknown>) {
  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  return resp.json();
}

const badgeColors = [
  "bg-primary",
  "bg-danger",
  "bg-warning",
  "bg-success",
  "bg-secondary",
];

// 4以下ならその数字を返す
// 4以上なら、再帰的にrounderに入って0,1,2,3,4のどれかを返す。
function rounder(i: number): number {
  if (i < badgeColors.length) {
    return i;
  }
  return rounder(i);
}

// 入力欄に打った情報をJSONでポスト
function postItem() {
  const url = root.origin + "/predict";
  const nameInput: HTMLInputElement = document.getElementById("name");
  const modelInput: HTMLInputElement = document.getElementById("model");
  const data = {
    "name": nameInput.value,
    "model": modelInput.value,
  };
  let j = 0;
  postData(url, data)
    .then((pidList: string[]) => {
      console.log(pidList); // DEBUG
      const resultDiv = document.getElementById("result");
      resultDiv.innerHTML = "";
      const h4 = document.createElement("h4");
      h4.innerHTML = "AIが予測する品番カテゴリは次のいずれかです。";
      resultDiv.appendChild(h4);
      pidList.forEach((p: string, i: number) => {
        const badge = document.createElement("span");
        const j = rounder(i); // 0,1,2,3,4 のサイクリック
        badge.classList.add("badge", "rounded-pill", badgeColors[j]); // Bootstrap Badge
        badge.innerHTML = p;
        resultDiv.appendChild(badge);
      });
    })
    .catch((e: Error) => {
      console.error(e);
    });
}
