---
title: Credit Card Fraud Detection
emoji: 💳
colorFrom: red
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# 💳 Credit Card Fraud Detection - Interactive Demo

Live demo of a production-grade fraud detection system trained on 284,807 credit card transactions using AWS SageMaker.

🔗 **Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/Donald8585/fraud-detection-demo)  
🔗 **Production Pipeline**: [fraud-detection-mlops](https://github.com/Donald8585/fraud-detection-mlops)

## 🎯 Features

- **Real-time fraud detection** with pre-trained XGBoost model
- **Interactive Gradio interface** with realistic transaction scenarios
- **Risk level classification** (Low/Medium/High/Critical)
- **Production-ready model** trained on AWS SageMaker
- **99.9% accuracy** on test set

## 📊 About the Model

### Dataset
- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 28 PCA-transformed features + Time + Amount
- **Class Distribution**: Highly imbalanced (0.172% fraud)

### Model Architecture
- **Algorithm**: XGBoost (Gradient Boosting)
- **Training Platform**: AWS SageMaker
- **Optimization**: Hyperparameter tuning via SageMaker
- **Deployment**: Serverless inference (AWS Lambda + API Gateway)

### Performance Metrics
- **Accuracy**: 99.9%
- **AUC-ROC**: 0.98+
- **Optimized for**: Minimizing false positives in production

## 🎮 Demo Scenarios

The demo includes 8 realistic transaction scenarios:

**Legitimate Transactions (Low Risk):**
- Normal retail purchases
- Online shopping
- Gas station transactions
- Restaurant bills

**Fraudulent Transactions (High Risk):**
- Confirmed fraud cases from dataset
- Various confidence levels (40% - 95%)
- Real fraud patterns detected by the model

## 🛠️ Tech Stack

- **ML Framework**: XGBoost, Scikit-learn
- **UI Framework**: Gradio
- **Training**: AWS SageMaker
- **Deployment**: HuggingFace Spaces (CPU)
- **Language**: Python 3.10+

## 🔗 Related Projects

- **[fraud-detection-mlops](https://github.com/Donald8585/fraud-detection-mlops)** - Full AWS production pipeline with SageMaker, Lambda, and API Gateway
- **[paligemma-image-captioning](https://github.com/Donald8585/paligemma-image-captioning)** - Vision-language model fine-tuning
- **[HK Healthcare RAG Chatbot](https://huggingface.co/spaces/Donald8585/hk-healthcare-rag)** - Domain-specific LLM application

## 👤 Author

**Alfred So (So Chit Wai)**
- GitHub: [@Donald8585](https://github.com/Donald8585)
- LinkedIn: [alfred-so](https://linkedin.com/in/alfred-so)
- Portfolio: [5+ Live ML Demos](https://huggingface.co/Donald8585)

## 📝 License

This project is open source and available under the MIT License.

---

⭐ **Built with XGBoost, Gradio, and AWS SageMaker** ⭐
