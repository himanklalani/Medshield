"use client";
import React, { useState, useRef } from "react";
import { Upload, Brain, Info, X, FileImage, Loader2, BarChart3, Download, BookOpen } from "lucide-react";

export default function BrainTumorClassifier() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  const [history, setHistory] = useState([]);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) processFile(file);
  };

  const processFile = (file) => {
    if (!["image/jpeg", "image/png", "image/bmp"].includes(file.type)) {
      alert("Please upload a valid image file (JPEG, PNG, or BMP)");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert("File size must be less than 10MB");
      return;
    }
    setSelectedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
    setPrediction(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setPrediction({
        class: data.class,
        confidence: Number(data.confidence).toFixed(2),
        color: data.class.includes("Healthy") ? "text-green-600" : "text-rose-600",
      });
      setHistory((prev) => [
        {
          id: Date.now(),
          fileName: selectedFile.name,
          prediction: data.class,
          confidence: Number(data.confidence).toFixed(2),
          timestamp: new Date().toLocaleTimeString(),
        },
        ...prev.slice(0, 4),
      ]);
    } catch (error) {
      alert("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setPrediction(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDownloadReport = () => {
    const report = `
FEDERATED BRAIN TUMOR CLASSIFIER - ANALYSIS REPORT
Generated: ${new Date().toLocaleString()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMAGE INFORMATION
File Name: ${selectedFile?.name}
Upload Time: ${new Date().toLocaleTimeString()}

PREDICTION RESULTS
Classification: ${prediction?.class}
Confidence Score: ${prediction?.confidence}%

MODEL INFORMATION
Model Accuracy: 94.3% (Validated)
Training Method: Federated Learning
Data Sources: Multi-institutional MRI scans

DISCLAIMER
This tool is for research and educational purposes only.
Results should not be used for clinical diagnosis without
professional medical evaluation.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    `;
    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `brain-tumor-report-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-[#244032] font-sans">
      {/* Header */}
      <header className="bg-[#e7d5c2] border-b border-[#c8b294] shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-[#c8b294] p-3 rounded-xl">
                <Brain className="w-10 h-10 text-[#244032]" />
              </div>
              <div>
                <h1 className="text-3xl font-extrabold text-[#244032]">MedShield-FL</h1>
                <p className="text-sm text-[#5d6c62]">Advanced AI tumor detection powered by federated learning</p>
              </div>
            </div>
            <button
              onClick={() => setShowAbout(true)}
              className="flex items-center gap-2 px-5 py-2 text-[#244032] hover:bg-[#c8b294] rounded-lg transition font-medium"
            >
              <BookOpen className="w-5 h-5" />
              <span className="hidden sm:inline">About</span>
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-8">
            {/* Upload Section */}
            {!preview && (
              <div className="bg-[#e7d5c2] rounded-2xl shadow-xl p-10 border border-[#c8b294]">
                <h2 className="text-2xl font-bold text-[#244032] mb-5">Upload MRI Scan</h2>
                <div
                  onDrop={handleDrop}
                  onDragOver={(e) => e.preventDefault()}
                  className="border-3 border-dashed border-[#c8b294] rounded-2xl p-16 text-center hover:border-[#244032] transition bg-[#e7d5c2] cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="w-20 h-20 mx-auto text-[#244032] mb-5" />
                  <p className="text-lg font-medium text-[#244032] mb-2">
                    Drag and drop your MRI scan here
                  </p>
                  <p className="text-sm text-[#5d6c62] mb-6">or click to browse files</p>
                  <button className="px-8 py-4 bg-[#c8b294] text-[#244032] rounded-xl font-bold hover:bg-[#244032] hover:text-[#e7d5c2] transition">
                    Choose File
                  </button>
                  <p className="text-xs text-[#5d6c62] mt-5">
                    Supported formats: JPEG, PNG, BMP (Max 10MB)
                  </p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/bmp"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
            )}

            {/* Preview & Results */}
            {preview && (
              <div className="bg-[#e7d5c2] rounded-2xl shadow-xl p-10 border border-[#c8b294]">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-[#244032]">Scan Preview</h2>
                  <button onClick={handleClear} className="text-[#244032] hover:text-rose-600 transition">
                    <X className="w-7 h-7" />
                  </button>
                </div>
                <div className="grid md:grid-cols-2 gap-8">
                  <div className="bg-[#d9ccb9] rounded-xl p-5">
                    <img src={preview} alt="MRI Preview" className="w-full h-72 object-contain rounded-lg border-4 border-[#c8b294]" />
                    <p className="text-sm text-[#5d6c62] mt-3 text-center truncate">{selectedFile?.name}</p>
                  </div>
                  <div className="flex flex-col justify-center">
                    {!prediction && !loading && (
                      <div className="text-center space-y-5">
                        <FileImage className="w-24 h-24 mx-auto text-[#c8b294]" />
                        <p className="text-[#244032] text-lg">Ready for analysis</p>
                        <button
                          onClick={handlePredict}
                          className="w-full px-6 py-5 bg-[#244032] text-[#e7d5c2] rounded-xl font-bold hover:bg-[#c8b294] hover:text-[#244032] transition shadow-lg"
                        >
                          Run Prediction
                        </button>
                      </div>
                    )}

                    {loading && (
                      <div className="text-center space-y-5">
                        <Loader2 className="w-24 h-24 mx-auto text-[#244032] animate-spin" />
                        <p className="text-[#244032] font-medium">Analyzing scan...</p>
                        <div className="w-full bg-[#c8b294] rounded-full h-3">
                          <div className="bg-[#244032] h-3 rounded-full animate-pulse" style={{ width: "70%" }}></div>
                        </div>
                      </div>
                    )}

                    {prediction && (
                      <div className="space-y-6">
                        <div className="bg-[#e7d5c2] rounded-xl p-7 border border-[#c8b294]">
                          <p className="text-sm text-[#244032] mb-2">Prediction</p>
                          <h3 className={`text-3xl font-bold ${prediction.color} mb-4`}>{prediction.class}</h3>
                          <p className="text-sm text-[#244032] mb-2">Confidence Score</p>
                          <div className="flex items-center gap-4 mb-2">
                            <div className="flex-1 bg-[#c8b294] rounded-full h-4">
                              <div
                                className="bg-[#244032] h-4 rounded-full transition-all duration-1000"
                                style={{ width: `${prediction.confidence}%` }}
                              ></div>
                            </div>
                            <span className="text-2xl font-bold text-[#244032]">{prediction.confidence}%</span>
                          </div>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <button
                            onClick={handleClear}
                            className="px-5 py-4 border-2 border-[#c8b294] text-[#244032] rounded-xl font-medium hover:bg-[#c8b294] transition"
                          >
                            Try Another
                          </button>
                          <button
                            onClick={handleDownloadReport}
                            className="px-5 py-4 bg-[#244032] text-[#e7d5c2] rounded-xl font-medium hover:bg-[#c8b294] hover:text-[#244032] transition flex items-center justify-center gap-2"
                          >
                            <Download className="w-5 h-5" />
                            Report
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Model Performance (Solid Box) */}
            <div className="bg-[#e7d5c2] rounded-2xl shadow-xl p-8 text-[#244032] border border-[#c8b294]">
              <div className="flex items-start gap-5">
                <BarChart3 className="w-14 h-14 flex-shrink-0" />
                <div>
                  <h3 className="text-2xl font-bold mb-3">Model Performance</h3>
                  <p className="mb-4">
                    Federated learning on multi-institutional MRI scans for privacy and accuracy.
                  </p>
                  <div className="flex items-center gap-8">
                    <div>
                      <p className="text-4xl font-bold">94.3%</p>
                      <p className="text-sm">Accuracy</p>
                    </div>
                    <div>
                      <p className="text-4xl font-bold">98.1%</p>
                      <p className="text-sm">Precision</p>
                    </div>
                    <div>
                      <p className="text-4xl font-bold">96.7%</p>
                      <p className="text-sm">Recall</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            {history.length > 0 && (
              <div className="bg-[#e7d5c2] rounded-2xl shadow-xl p-7 border border-[#c8b294]">
                <h3 className="text-lg font-bold text-[#244032] mb-4">Recent Analysis</h3>
                <div className="space-y-4">
                  {history.map((item) => (
                    <div key={item.id} className="bg-[#d9ccb9] rounded-lg p-4 border border-[#c8b294]">
                      <p className="text-sm font-medium text-[#244032] truncate">{item.fileName}</p>
                      <p className="text-xs text-[#5d6c62] mt-1">{item.prediction}</p>
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-xs text-[#5d6c62]">{item.timestamp}</span>
                        <span className="text-xs font-bold text-[#244032]">{item.confidence}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="bg-[#e7d5c2] rounded-2xl shadow-xl p-7 border border-[#c8b294]">
              <h3 className="text-lg font-bold text-[#244032] mb-4">Classification Types</h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 bg-[#d9ccb9] rounded-lg border border-[#c8b294]">
                  <div className="w-3 h-3 bg-black-000 rounded-full"></div>
                  <span className="text-sm font-medium text-[#244032]">Healthy Brain</span>
                </div>
                <div className="flex items-center gap-3 p-3 bg-[#d9ccb9] rounded-lg border border-[#c8b294]">
                  <div className="w-3 h-3 bg-black-500 rounded-full"></div>
                  <span className="text-sm font-medium text-[#244032]">Brain Glioma</span>
                </div>
                <div className="flex items-center gap-3 p-3 bg-[#d9ccb9] rounded-lg border border-[#c8b294]">
                  <div className="w-3 h-3 bg-black-500 rounded-full"></div>
                  <span className="text-sm font-medium text-[#244032]">Brain Meningioma</span>
                </div>
                <div className="flex items-center gap-3 p-3 bg-[#d9ccb9] rounded-lg border border-[#c8b294]">
                  <div className="w-3 h-3 bg-black-500 rounded-full"></div>
                  <span className="text-sm font-medium text-[#244032]">Pituitary Tumor</span>
                </div>
              </div>
            </div>

            {/* Disclaimer box remains in amber for visibility */}
            <div className="bg-amber-50 rounded-2xl shadow-xl p-7 border border-amber-200">
              <div className="flex gap-3">
                <Info className="w-6 h-6 text-amber-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-bold text-amber-900 mb-2">Medical Disclaimer</h3>
                  <p className="text-xs text-amber-800 leading-relaxed">
                    This tool is for research and educational purposes only. Results should not be used for clinical diagnosis without professional medical evaluation.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* About Modal */}
      {showAbout && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center p-4 z-50">
          <div className="bg-[#e7d5c2] rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-[#c8b294]">
            <div className="sticky top-0 bg-[#e7d5c2] border-b border-[#c8b294] px-6 py-5 flex items-center justify-between">
              <h2 className="text-2xl font-bold text-[#244032]">About NeuroShield AI</h2>
              <button onClick={() => setShowAbout(false)} className="text-[#244032] hover:text-[#c8b294] transition">
                <X className="w-6 h-6" />
              </button>
            </div>
            <div className="p-6 space-y-6">
              <div>
                <h3 className="text-lg font-bold text-[#244032] mb-2">Federated Learning Model</h3>
                <p className="text-[#244032] leading-relaxed">
                  This system uses a deep learning model trained using federated learning—a privacy-preserving method where the model learns from data distributed across multiple hospitals without ever leaving those institutions. This enhances patient privacy while enabling more robust and collaborative AI development.
                </p>
              </div>
              <div>
                <h3 className="text-lg font-bold text-[#244032] mb-2">Key Benefits</h3>
                <ul className="list-disc list-inside text-[#244032] space-y-2">
                  <li>Ensures patient data privacy; data never leaves hospitals</li>
                  <li>Model learns from diverse MRI scan sources for better generalization</li>
                  <li>Compliant with healthcare data regulations (HIPAA, GDPR)</li>
                  <li>Supports decision making for healthcare professionals</li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-bold text-[#244032] mb-2">Model Architecture</h3>
                <p className="text-[#244032] leading-relaxed">
                  The classifier is built on a convolutional neural network (CNN) architecture optimized for high-accuracy medical imaging analysis. The model was validated using thousands of MRI scans from multiple medical centers.
                </p>
              </div>
              <div>
                <h3 className="text-lg font-bold text-[#244032] mb-2">Limitations & Usage</h3>
                <ul className="list-disc list-inside text-[#244032] space-y-2">
                  <li>For research and demonstration only</li>
                  <li>Not a substitute for professional medical diagnosis</li>
                  <li>Image quality and scanning protocols may affect prediction accuracy</li>
                  <li>To be used as a decision support tool, not final diagnosis</li>
                </ul>
              </div>
              <div className="bg-[#d9ccb9] rounded-xl p-4 border border-[#c8b294]">
                <h3 className="text-lg font-bold text-[#244032] mb-2">Technical Specifications</h3>
                <div className="text-sm text-[#244032] space-y-1">
                  <p><span className="font-bold">Model Type:</span> CNN (Convolutional Neural Network)</p>
                  <p><span className="font-bold">Training Method:</span> Federated Learning</p>
                  <p><span className="font-bold">Test Accuracy:</span> 94.3%</p>
                  <p><span className="font-bold">Classes:</span> Healthy, Glioma, Meningioma, Pituitary Tumor</p>
                  <p><span className="font-bold">Input:</span> MRI Brain Scans (JPEG, PNG, BMP)</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
