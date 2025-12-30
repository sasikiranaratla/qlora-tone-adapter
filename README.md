# CounselorLlama: Domain-Specific Fine-Tuning of Llama-3.2-3B

**[Try on Google Colab](https://colab.research.google.com/drive/14SF_bVJB1SktFnPOdK8AJKPN-dQwGTK-?usp=sharing)** | [Model Weights](https://huggingface.co/sasikiranaratla/llama-3.2-3b-counselor) | [Dataset](https://huggingface.co/datasets/sasikiranaratla/proverbial-counselor-data)

## Project Overview

A specialized Small Language Model (SLM) designed for mental health supportive dialogue, fine-tuned on Llama-3.2-3B to enforce a structured "Proverbial Counseling" architecture:

- **Ancient Metaphor:** Opens responses with contextually relevant proverbs
- **Empathetic Mirroring:** Validates emotional states using clinical terminology  
- **Socratic Inquiry:** Closes with discovery questions to facilitate self-reflection

**Key Achievements:**
- 2x faster training via Unsloth optimization & 60% memory reduction (4-bit NF4 quantization)
- Curated 370-sample dataset with structural validation (regex & semantic analysis)
- Deployed on HuggingFace with production-ready inference

---

## Technical Stack

**Models & Libraries:** Llama-3.2-3B, Unsloth, PyTorch, QLoRA, Hugging Face Transformers  
**Training Optimization:** 4-bit NF4 Quantization, LoRA (r=16, α=32), AdamW Optimizer  
**Data Engineering:** 370-sample ChatML dataset, regex-based structural validation  
**Experiment Tracking:** Weights & Biases (W&B) for metrics & gradient monitoring  
**Deployment:** HuggingFace Model Hub with production-ready inference

---

## Usage

**Try on Google Colab** (recommended): Click the Colab link at the top to run inference with free GPU access. The notebook includes examples of the model's structured output and supports fine-tuning experiments.

**Local Inference:** Use the model weights from HuggingFace with the Unsloth library for optimized inference:

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("sasikiranaratla/llama-3.2-3b-counselor")
FastLanguageModel.for_inference(model)

# The model enforces Proverb → Validation → Question structure
```

---

## Project Details

- **Training Data:** 370 curated counseling dialogues in ChatML format
- **Validation:** Regex-based structural validation ensures model adheres to the 3-part architecture
- **Performance:** Optimized for low-latency edge deployment while maintaining reasoning quality
- **Resources:** Experiment tracking via Weights & Biases for reproducibility

---

## Quick Links

- [Model on HuggingFace](https://huggingface.co/sasikiranaratla/llama-3.2-3b-counselor)
- [Training Dataset](https://huggingface.co/datasets/sasikiranaratla/proverbial-counselor-data)
- [Open in Google Colab](https://colab.research.google.com/drive/14SF_bVJB1SktFnPOdK8AJKPN-dQwGTK-?usp=sharing)
