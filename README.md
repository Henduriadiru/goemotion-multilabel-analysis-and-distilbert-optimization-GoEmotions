# goemotion-multilabel-analysis-and-distilbert-optimization-GoEmotions-ONNX

# GoEmotions Multi-Label Emotion Classification with DistilBERT, LoRA, Optuna, and ONNX

This repository presents a complete deep learning pipeline for multi-label emotion classification using the [GoEmotions](https://github.com/google-research/goemotions) dataset. All modeling and training steps are implemented in **pure PyTorch**, without relying on HuggingFace's Trainer or pipeline. HuggingFace Transformers are used only for loading pretrained models and tokenizers.

---

## 📂 Repository Structure
📁 goemotion-distilbert-multilabel-optuna-onnx
├── 1_analysis_goemotion_multilabel.ipynb # EDA & label binarization
├── 2_distilbert_lora_multilabel_optuna.ipynb # Custom PyTorch training with LoRA and Optuna
└── (Coming Soon) 3_export_onnx_distilbert.ipynb # ONNX export & benchmarking

---

## ✅ Highlights

- 🔍 Multi-label classification using **DistilBERT**
- 🧠 Custom model and training loop using **PyTorch**
- 🔧 Parameter-efficient fine-tuning using **LoRA**
- 🔬 Hyperparameter optimization with **Optuna**
- 📊 Comprehensive evaluation with F1, precision, recall
- 🚀 Planned: Export and inference using **ONNX**

---

## 📌 Why This Project is Different

- **No HuggingFace Trainer used** – All training logic is implemented with native PyTorch, offering full control.
- **LoRA** is implemented to reduce training time and memory.
- **Optuna** is used to automatically find the best hyperparameters.
- Designed for reproducibility and adaptation to other multi-label tasks.

---

## 📈 Evaluation Metrics

- Micro-averaged and macro-averaged precision, recall, and F1
- Label-wise classification report
- Samples-averaged accuracy for multilabel prediction

---

## 📊 Preliminary Evaluation Summary

**Test Set Results (Before Final Tuning & ONNX):**
- **Micro F1-score:** `0.54`
- **Macro F1-score:** `0.37`
- **Weighted F1-score:** `0.50`
- **Samples Accuracy:** `0.47`

**Top Performing Emotions:**
- `Gratitude` — F1: **0.91**
- `Amusement` — F1: **0.81**
- `Love` — F1: **0.80**
- `Admiration` — F1: **0.65**
- `Neutral` — F1: **0.62**

**Challenging Emotions (due to imbalance or label confusion):**
- `Grief`, `Relief`, `Nervousness`, `Realization` — F1 near `0.00`

> 📌 Note: These are early results using default hyperparameters and a basic LoRA setup. Further tuning with Optuna and ONNX optimization is in progress.

## ⚙️ Tech Stack

- PyTorch (`torch`)
- HuggingFace `transformers` (only for model/tokenizer loading)
- peft (for efficiency tunning using LoRA)
- Optuna (for tuning learning rate, LoRA rank, etc.)
- scikit-learn
- Pandas / Seaborn / Matplotlib
- ONNX (planned for export)

---

## 📊 Planned Work

- ✅ Finish model training with best Optuna parameters
- ✅ Generate and save classification report
- 🔜 Export best model to **ONNX** for lightweight deployment
- 🔜 Benchmark ONNX inference (CPU/GPU)

---

## 📬 Feedback & Collaboration

Open an issue or fork the repo if you’re interested in collaborating, improving LoRA handling, or helping with ONNX deployment.

---

⭐ If this repo helps your research or project, please consider giving it a star!
