import { useState } from "react";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();

    formData.append("image", selectedFile);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data = await response.json();

      setResult(data);
    } catch (err) {
      setError(
        "Unable to connect to the prediction server. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">

        <h1>Plant Disease Detection</h1>

        <p className="subtitle">
          Upload a tomato leaf image to detect its condition.
        </p>

        <div className="upload-section">

          <label className="file-label">
            Select Image
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
            />
          </label>

          {selectedFile && (
            <p className="filename">
              Selected: {selectedFile.name}
            </p>
          )}

        </div>

        {preview && (
          <div className="preview-section">

            <h2>Image Preview</h2>

            <img
              src={preview}
              alt="Selected tomato leaf"
              className="preview-image"
            />

          </div>
        )}

        <button
          className="predict-button"
          onClick={handlePredict}
          disabled={loading || !selectedFile}
        >
          {loading ? "Predicting..." : "Predict Disease"}
        </button>

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {result && (
          <div className="result-card">

            <h2>Prediction Result</h2>

            <p>
              <strong>Condition:</strong>
            </p>

            <div className="prediction">
              {result.prediction}
            </div>

            <p>
              <strong>Confidence:</strong>
            </p>

            <div className="confidence">
              {(result.confidence * 100).toFixed(2)}%
            </div>

          </div>
        )}

      </div>
    </div>
  );
}

export default App;