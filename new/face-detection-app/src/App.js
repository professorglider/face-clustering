import React, { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";
import "./App.css";

async function loadModels() {
  const MODEL_URL = "/weights";
  await faceapi.loadTinyFaceDetectorModel(MODEL_URL);
  await faceapi.loadFaceLandmarkModel(MODEL_URL);
  await faceapi.loadFaceRecognitionModel(MODEL_URL);
  await faceapi.loadFaceExpressionModel(MODEL_URL);
}

function Webcam() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [embeddings, setEmbeddings] = useState([]);
  const [expressions, setExpressions] = useState([]);

  useEffect(() => {
    loadModels().then(() => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then((stream) => {
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
            }
          })
          .catch((error) => console.error("Error accessing webcam: ", error));
      } else {
        console.error("getUserMedia not supported in this browser");
      }
    });

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(async () => {
      if (videoRef.current && canvasRef.current && faceapi.nets.tinyFaceDetector.params) {
        const displaySize = { width: videoRef.current.width, height: videoRef.current.height };
        faceapi.matchDimensions(canvasRef.current, displaySize);
        const detections = await faceapi
          .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptors()
          .withFaceExpressions();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        const ctx = canvasRef.current.getContext("2d");
        ctx.clearRect(0, 0, displaySize.width, displaySize.height);
        faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections);

        const newEmbeddings = resizedDetections.map((detection) => detection.descriptor);
        setEmbeddings(newEmbeddings);

        const newExpressions = resizedDetections.map((detection) => detection.expressions);
        setExpressions(newExpressions);
      }
    }, 50);

    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="App">
      <div className="video-container">
        <video ref={videoRef} autoPlay muted playsInline width="720" height="560" />
        <canvas ref={canvasRef} width="720" height="560" />
      </div>
      <div className="embeddings">
        <h2>Face Embeddings:</h2>
        {embeddings.map((embedding, index) => (
          <div key={index}>
            <strong>Face {index + 1}:</strong>{" "}
            {embedding.map((value) => value.toFixed(4)).join(", ")}
          </div>
        ))}
      </div>
      <div className="expressions">

        <h2>Emotion Recognition:</h2>
        {expressions.map((expression, index) => (
          <div key={index}>
            <strong>Face {index + 1}:</strong>{" "}
            {Object.entries(expression)
              .sort(([, a], [, b]) => b - a)
              .map(([key, value], i) => (
                <span key={i}>
                  {key}: {value.toFixed(2)}
                  {i < Object.entries(expression).length - 1 ? ", " : ""}
                </span>
              ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  return <Webcam />;
}
