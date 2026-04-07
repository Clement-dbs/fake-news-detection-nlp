const API_BASE_URL = "http://127.0.0.1:8000";
const PREDICT_URL = `${API_BASE_URL}/predict`;

const form = document.getElementById("prediction-form");
const resultCard = document.getElementById("result-card");
const resultContainer = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = document.getElementById("message").value;

  try {

    const response = await fetch(PREDICT_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: message
      })
    });

    if (!response.ok) {
      throw new Error("Erreur lors de l'appel à l'API");
    }

    const data = await response.json();
    renderResult(data);

  } catch (error) {
    resultCard.classList.remove("hidden");
    resultContainer.innerHTML = `<div class="result-item">Erreur : ${error.message}</div>`;
  } 
});


function renderResult(data) {
  resultCard.classList.remove("hidden");

  const scoresHtml = Object.entries(data.scores)
    .sort((a, b) => b[1] - a[1])
    .map(([label, score]) => `
      <div class="score-item">
        <strong>${label}</strong> : ${(score * 100).toFixed(2)} %
      </div>
    `)
    .join("");

  resultContainer.innerHTML = `
    <div class="result-grid">
      <div class="result-item"><strong>Texte :</strong> ${data.text}</div>
      <div class="result-item"><strong>Classement :</strong> <span class="label-badge">${data.predicted_label}</span></div>
      <div class="result-item"><strong>Confiance :</strong> ${(data.confidence * 100).toFixed(2)} %</div>

    </div>
    <h3>Scores détaillés</h3>
    <div class="score-list">${scoresHtml}</div>
  `;
}



