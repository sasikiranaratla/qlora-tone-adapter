# 🌿 CounselorLlama: Domain-Specific Fine-Tuning of Llama-3.2-3B

[![Model: Llama-3.2-3B](https://img.shields.io/badge/Model-Llama--3.2--3B-blue)](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
[![Library: Unsloth](https://img.shields.io/badge/Library-Unsloth-orange)](https://github.com/unslothai/unsloth)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)
[![Open in Colab](https://colab.research.google.com/drive/14SF_bVJB1SktFnPOdK8AJKPN-dQwGTK-?usp=sharing)

## 🚀 Quick Start with Google Colab
Want to try CounselorLlama without local setup? Click the **"Open in Colab"** badge above to run the model directly in your browser using Google Colab's free GPU. This notebook provides a complete interactive environment for inference, fine-tuning, and experimentation with the model. No installation required—just authenticate with your Google account and start exploring!

## 📌 Project Overview
CounselorLlama is a specialized **Small Language Model (SLM)** designed for mental health supportive dialogue. While general-purpose LLMs provide direct answers, this model is fine-tuned to follow a strict **"Proverbial Counseling"** architecture. 

The primary objective was to engineer a model that enforces a specific structural constraint:
1. **Ancient Metaphor:** Begin every response with a contextually relevant proverb.
2. **Empathetic Mirroring:** Validate the user's emotional state using clinical terminology.
3. **Socratic Inquiry:** Conclude with a clinical discovery question to facilitate self-reflection.

---

## � Model, Dataset & Monitoring
* **Model (trained weights):** https://huggingface.co/sasikiranaratla/llama-3.2-3b-counselor
* **Dataset (training data):** https://huggingface.co/datasets/sasikiranaratla/proverbial-counselor-data
* **Experiment Tracking & Analysis:** We used **Weights & Biases (W&B)** to monitor, log, and analyze fine-tuning runs (metrics, gradients, and artifacts).

---

## �🛠 Technical Stack & Engineering Decisions

### 1. Model Selection: Llama-3.2-3B
* **Rationale:** I chose the 3B parameter variant over larger models (e.g., 70B) to optimize for **edge deployment** and **low-latency inference**. It provides sufficient reasoning density for structured tasks while remaining small enough for mobile or consumer-grade hardware.

### 2. Efficiency: Unsloth & QLoRA
* **Quantization:** Implemented **4-bit NormalFloat (NF4)** quantization to fit the model within 16GB VRAM (NVIDIA T4).
* **Speed:** Utilized **Unsloth's optimized kernels**, which resulted in a **2x faster training time** and **60% less memory usage** compared to standard Hugging Face PEFT.

### 3. Hyperparameter Configuration
| Parameter | Value | Engineering Reason |
| :--- | :--- | :--- |
| **LoRA Rank (r)** | 16 | Chosen to provide enough capacity to learn the 3-part structure without "catastrophic forgetting." |
| **LoRA Alpha** | 32 | Follows the $2 \times r$ rule for stable gradient updates. |
| **Learning Rate** | 2e-4 | The "goldilocks" rate for QLoRA to ensure convergence on small datasets. |
| **Global Batch Size** | 16 | Achieved via 4 accumulation steps to stabilize training signals from 300+ samples. |
| **Warmup Ratio** | 0.1 | 10% warmup ensures the AdamW optimizer adapts to the custom persona gradually. |

---

## 📊 Data Engineering & MLOps
* **Curated Dataset:** Engineered a specialized 370-sample dataset in `ChatML` format.
* **Experiment Tracking:** Integrated **Weights & Biases (W&B)** to monitor `eval/loss` and `gradient_norm`.
* **Validation Strategy:** Implemented a structural validation script that uses regex and semantic analysis to verify the model adheres to the Proverb-Validation-Question format on unseen inputs.



---

## 🚀 Deployment & Usage

### Local Inference
The model is optimized for low-latency generation. Using Unsloth's inference engine:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("sasikiranaratla/llama-3.2-3b-counselor")
FastLanguageModel.for_inference(model)

# The model strictly follows the Llama-3.2 Chat Template
