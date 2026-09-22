const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const dropZonePlaceholder = document.getElementById("drop-zone-placeholder");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");
const resetBtn = document.getElementById("reset-btn");
const loadingEl = document.getElementById("loading");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const resultBadge = document.getElementById("result-badge");
const resultConfidence = document.getElementById("result-confidence");
const resultRaw = document.getElementById("result-raw");

const MAX_SIZE = 5 * 1024 * 1024;
const ALLOWED_TYPES = ["image/png", "image/jpeg"];

function resetOutputs() {
  errorEl.hidden = true;
  errorEl.textContent = "";
  resultEl.hidden = true;
}

function showError(mensaje) {
  resetOutputs();
  errorEl.hidden = false;
  errorEl.textContent = mensaje;
}

function setFile(file) {
  if (!file) return;

  if (!ALLOWED_TYPES.includes(file.type)) {
    showError("Formato no soportado. Usa una imagen PNG o JPG.");
    return;
  }

  if (file.size > MAX_SIZE) {
    showError("La imagen supera el tamaño máximo de 5 MB.");
    return;
  }

  resetOutputs();

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropZonePlaceholder.hidden = true;
  };
  reader.readAsDataURL(file);

  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  fileInput.files = dataTransfer.files;

  submitBtn.disabled = false;
  resetBtn.hidden = false;
}

fileInput.addEventListener("change", () => {
  setFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((evento) => {
  dropZone.addEventListener(evento, (e) => {
    e.preventDefault();
    dropZone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((evento) => {
  dropZone.addEventListener(evento, (e) => {
    e.preventDefault();
    dropZone.classList.remove("is-dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  setFile(file);
});

resetBtn.addEventListener("click", () => {
  fileInput.value = "";
  preview.src = "";
  preview.hidden = true;
  dropZonePlaceholder.hidden = false;
  submitBtn.disabled = true;
  resetBtn.hidden = true;
  resetOutputs();
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!fileInput.files.length) {
    showError("Selecciona una imagen primero.");
    return;
  }

  resetOutputs();
  loadingEl.hidden = false;
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append("imagen", fileInput.files[0]);

  try {
    const respuesta = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const datos = await respuesta.json();

    if (!respuesta.ok) {
      throw new Error(datos.error || "Ocurrió un error al analizar la imagen.");
    }

    const esNormal = datos.resultado.toLowerCase() === "normal";

    resultBadge.textContent = datos.resultado;
    resultBadge.className = "badge " + (esNormal ? "badge--normal" : "badge--pneumonia");
    resultConfidence.textContent = `Confianza: ${datos.confianza}%`;
    resultRaw.textContent = `Valor bruto del modelo: ${datos.probabilidad_bruta}`;
    resultEl.hidden = false;
  } catch (err) {
    showError(err.message);
  } finally {
    loadingEl.hidden = true;
    submitBtn.disabled = false;
  }
});
