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
      resultDiv.innerHTML = "";
      const h4 = document.createElement("h4");
      h4.innerHTML = "AIが予測する品番カテゴリは次のいずれかです。";
      resultDiv.appendChild(h4);
      pidList.forEach((p: string) => {
        const badge = document.createElement("span");
        badge.classList.add("badge", "rounded-pill", "bg-primary"); // Bootstrap Badge
        badge.innerHTML = p;
        resultDiv.appendChild(badge);
      });
    })
    .catch((e: Error) => {
      console.error(e);
    });
}
