# 🎯 PatternHunter — Design Pattern Detection with ML

> Automatically detect which Design Pattern is implemented in a Java code snippet using Machine Learning.

---

## 🎯 Objective

Given a Java code snippet, predict which of the **8 Design Patterns** is implemented:

`Singleton` · `Abstract Factory` · `Factory Method` · `Strategy` · `Observer` · `Adapter` · `Decorator` · `Facade`

---

## 📊 3 Models Compared

| # | Approach | Description | Time (GPU T4) | VRAM | Model Size |
|---|----------|-------------|:-------------:|:----:|:----------:|
| 1 | **TF-IDF + SVM** | Classic, fast, interpretable | < 1 min | ❌ No GPU | ~10 MB |
| 2 | **CodeBERT Fine-tuning** | Code-specialized Transformer (125M params) | ~30 min | ~6 GB | ~500 MB |
| 3 | **LoRA Fine-tuning (PEFT)** | CodeBERT + LoRA, only ~1% of weights trained | ~20 min | ~3 GB | ~10 MB |

---

## 🏆 Results (Test Set — 400 samples)

| Approach | Accuracy | F1 Macro | F1 Weighted | Time |
|----------|:--------:|:--------:|:-----------:|:----:|
| TF-IDF + SVM | 100.00% | 100.00% | 100.00% | < 1 min |
| CodeBERT Fine-tuning | 100.00% | 100.00% | 100.00% | ~5 min* |
| LoRA Fine-tuning | 100.00% | 100.00% | 100.00% | ~5 min* |

*Measured on Google Colab with Tesla T4 GPU*

---

## 📁 Project Structure

```
├── compare_3_approaches.ipynb   # Main notebook
├── data/
│   ├── singleton_java_500.jsonl
│   ├── abstract_factory_java_500.jsonl
│   ├── factory_method_java_500.jsonl
│   ├── strategy_java_500.jsonl
│   ├── observer_java_500.jsonl
│   ├── adapter_java_500.jsonl
│   ├── decorator_java_500.jsonl
│   └── facade_java_500.jsonl
├── model_svm.pkl                # Saved TF-IDF + SVM pipeline
├── model_codebert/              # Fine-tuned CodeBERT
├── model_lora/                  # LoRA adapters (~10 MB)
└── comparison_results.csv       # Comparative results table
```

---

## 🚀 Getting Started

### Requirements

```bash
pip install transformers datasets accelerate peft scikit-learn seaborn torch
```

### On Google Colab

1. `Runtime` → `Change runtime type` → **GPU T4**
2. Upload your 8 `.jsonl` files into `/content/data/`
3. `Runtime` → `Run all`

---

## 🔧 Technical Details

### Approach 1 — TF-IDF + SVM

- Java identifier tokenization (`[a-zA-Z_][a-zA-Z0-9_]*`)
- N-grams 1 to 3, max 50,000 features, `sublinear_tf=True`
- Calibrated `LinearSVC` with 3-fold cross-validation

### Approach 2 — CodeBERT Full Fine-tuning

- Model: `microsoft/codebert-base` (125M parameters)
- 5 epochs, batch size 16, lr `2e-5`, cosine scheduler
- Early stopping (patience=2)
- Mixed precision FP16

### Approach 3 — LoRA (PEFT)

- Base model: `microsoft/codebert-base` (99% frozen)
- LoRA rank: `r=16`, `lora_alpha=32`, dropout `0.1`
- Targets: `query` and `value` attention layers
- 8 epochs, lr `3e-4`, early stopping (patience=3)

```
LoRA : W_new = W_original + (A × B)
       W_original is frozen, only A and B are trained
       rank(A×B) << rank(W)
```

---

## 📋 Decision Guide — Which Approach to Choose?

| Criterion | TF-IDF + SVM | CodeBERT | LoRA |
|-----------|:---:|:---:|:---:|
| Accuracy (F1) | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Training speed | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| VRAM required | ❌ | ~6 GB | ~3 GB |
| Real-world generalization | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Interpretability | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ |
| Model size | ~10 MB | ~500 MB | ~10 MB |

### ✅ Recommendations

- **Quick prototype / no GPU** → TF-IDF + SVM
- **Best absolute accuracy** → CodeBERT fine-tuning
- **Best speed/accuracy tradeoff** → LoRA *(recommended for production)*
- **Lightweight deployment** → LoRA *(save only the adapters ~10 MB)*

---

## 🧹 Code Preprocessing

A `clean_code()` function is applied to all samples before training to remove synthetic artifacts:

```python
def clean_code(code):
    code = re.sub(r'//\s*Style:.*\n', '', code)           # remove style comments
    code = re.sub(r'(\b[A-Za-z][A-Za-z0-9_]*)(\d+)\b', r'\1', code)  # ConfigManager42 → ConfigManager
    code = re.sub(r'\n{3,}', '\n\n', code)
    return code.strip()
```

---

## 📦 Dataset

- **4,000 samples** total (500 per pattern)
- Stratified split: **80% train / 10% val / 10% test**
- Format: `.jsonl` files with a `code` field

---

## 📄 License

MIT
