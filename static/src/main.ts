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
  const modelInput: HTMLInputElement = document.getElementById("model");
  const data = {
    "name": nameInput.value,
    "model": modelInput.value,
  };
  postData(url, data)
    .then((pidList: string[]) => {
      console.log(pidList); // DEBUG
      const resultDiv = document.getElementById("result");
      resultDiv.innerHTML = ""; // Reset result div
      const h4 = document.createElement("h4");
      h4.innerHTML = "AIが予測する品番カテゴリは次のいずれかです。";
      resultDiv.appendChild(h4);
      pidList.forEach((p: string, i: number) => {
        const badge = document.createElement("button");
        badge.setAttribute("type", "button");
        badge.classList.add("badge", "rounded-pill", badgeSelector(i)); // Bootstrap Badge
        badge.innerHTML = p; // PID カテゴリ
        resultDiv.appendChild(badge);
      });
    })
    .catch((e: Error) => {
      console.error(e);
    });
}
