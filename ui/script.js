document.getElementById("predict-form").onsubmit = async function(event) {
  event.preventDefault();
  const formData = new FormData(event.target);
  const features = {};
  formData.forEach((value, key) => {
    if(key !== "model_type") {
      features[key] = parseFloat(value);
    }
  });
  const modelType = document.getElementById("model_type").value;

  const response = await fetch("/predict/?model_type=" + modelType, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features)
  });
  const result = await response.json();
  document.getElementById("result").innerText = 
    "Predicted Energy Consumption (MWh): " + Number(result.prediction).toFixed(2);
};
