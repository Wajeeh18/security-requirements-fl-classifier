# Security Requirements FL Classifier (SR vs NSR)

A privacy-preserving text classification pipeline that identifies **Security Requirements (SR)** vs. **Non-Security Requirements (NSR)** from software requirement specifications, using a fine-tuned RoBERTa model trained via **Federated Learning (FedAvg)**, followed by centralized fine-tuning and full explainability analysis (SHAP + LIME).

---

## 📌 Overview

Software requirement documents contain a mix of functional and non-functional requirements, and only some of them relate to security. Manually flagging security-relevant requirements is slow and inconsistent at scale. This project automates that classification using a transformer-based NLP model, while simulating a **federated learning** setup where training happens across multiple decentralized data "clients" without centralizing raw data. This is followed by a short centralized fine-tuning phase to close the remaining performance gap.

The pipeline also includes model interpretability using SHAP and LIME, so predictions aren't a black box. You can see exactly which words pushed a requirement toward "Security" or "Non-Security."

---

## 🧠 Approach

1. **Data preparation.** Synthetic data combined with an original requirements dataset (SecReq + DOSSPre / PROMISE-style sources), cleaned and mapped to binary labels: `SR` (1) and `NSR` (0).
2. **Model.** `roberta-base` fine-tuned for binary sequence classification, with the bottom N transformer layers frozen for stability and efficiency.
3. **Federated Learning simulation.** Training data is split across 3 simulated clients. Each round, clients train locally, and their weights are aggregated on the server using **FedAvg**. Global test accuracy is tracked after every round.
4. **Centralized fine-tuning.** After FL rounds plateau, a short centralized fine-tune (low learning rate) on the full training set closes the remaining accuracy gap.
5. **Evaluation.** Accuracy, precision, recall, F1 (overall and per-class), ROC curves, precision-recall curves, and confusion matrix.
6. **Explainability (XAI).** SHAP token-importance plots and waterfall plots, plus LIME explanations for individual predictions, saved as both PNG and interactive HTML.
7. **Inference.** A ready-to-use `predict()` function classifies new requirement text and optionally explains the prediction.

---

## 📂 Dataset

The dataset used for this project is already included in the `dataset/` folder of this repository.

**Input format:** CSV files (`merged_train.csv`, `merged_test.csv`) with at least two columns:
- `Requirement`: the requirement text
- `Class`: label, either `SR` or `NSR`

Text is cleaned (whitespace normalization, stripped quotes, special character removal) before tokenization. If you move the dataset files elsewhere, update `TRAIN_PATH` and `TEST_PATH` in the notebook's config section to match.

---

## ⚙️ Configuration

Key hyperparameters used in this pipeline (adjustable in the config section of the notebook):

| Parameter | Value |
|---|---|
| Base model | `roberta-base` |
| Max sequence length | 128 |
| Batch size (micro / effective) | 16 / 64 (via gradient accumulation) |
| Epochs per FL round | 3 |
| Learning rate | 2e-5 |
| Frozen layers | 4 (bottom transformer layers) |
| FL clients | 3 |
| FL rounds | 8 |
| Fine-tune epochs (post-FL) | 2 |
| Fine-tune learning rate | 5e-6 |
| Mixed precision (AMP) | Enabled when GPU is available |

---

## 🛠️ Tools & Libraries

- `transformers` (RoBERTa tokenizer and sequence classification model)
- `torch` (training, mixed precision via `autocast`/`GradScaler`)
- `scikit-learn` (metrics: accuracy, F1, ROC/PR curves, confusion matrix)
- `shap`, `lime` (explainability)
- `flwr` (Flower, federated learning simulation framework)
- `pandas`, `numpy`, `matplotlib`, `seaborn` (data handling and visualization)
- `tqdm` (progress bars)

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install transformers datasets torch scikit-learn shap lime
   pip install flwr[simulation] matplotlib seaborn pandas
   ```
2. Make sure the dataset files (`merged_train.csv`, `merged_test.csv`) are available at the paths configured in the notebook (default: `/content/merged_train.csv`, `/content/merged_test.csv`, update if running outside Colab).
3. Run the notebook top to bottom. It will:
   - Load and tokenize the data
   - Run federated training across simulated clients
   - Fine-tune the global model centrally
   - Evaluate and generate all figures
   - Run SHAP and LIME explainability
   - Save the final model to `security_req_roberta/`
4. Use the `predict()` function at the end to classify new requirement text:
   ```python
   predict("The system shall encrypt data in transit using TLS 1.3.", explain=True)
   ```

---

## 📁 Repository Structure

```
security-requirements-fl-classifier/
│
├── Security_Requirements_Classification.ipynb   # Main notebook
├── dataset/                                      # merged_train.csv, merged_test.csv
└── README.md                                     # This file
```

---

## 🤝 Contributing

Suggestions, bug reports, and pull requests are welcome, especially around dataset expansion, additional client-heterogeneity experiments, or alternative aggregation strategies beyond FedAvg.
