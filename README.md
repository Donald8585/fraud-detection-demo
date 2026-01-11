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

## 🚀 Quick Start

### Local Deployment

```bash
# Clone the repository
git clone https://github.com/Donald8585/fraud-detection-demo.git
cd fraud-detection-demo

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### HuggingFace Spaces Deployment

1. Create new Space on HuggingFace
2. Upload all files to the Space
3. Set SDK to "Gradio"
4. Space will auto-deploy!

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
- **Precision**: High (minimizes false positives)
- **Recall**: Optimized for fraud detection
- **F1-Score**: Balanced performance

## 🎮 Demo Scenarios

The demo includes 8 realistic transaction scenarios:

**Legitimate Transactions:**
- Normal grocery purchase ($45.80)
- Small online purchase ($23.50)
- Gas station ($67.20)
- Restaurant bill ($156.40)

**Fraudulent Patterns:**
- Large unusual transaction ($8,500)
- Suspicious international transaction ($1,250)
- Rapid multiple transactions ($999.99 x 3) - Structuring
- Late night high-value ($3,200)

## 🛠️ Tech Stack

- **ML Framework**: XGBoost, Scikit-learn
- **UI Framework**: Gradio
- **Training**: AWS SageMaker
- **Deployment**: HuggingFace Spaces (CPU)
- **Language**: Python 3.10+

## 📁 Project Structure

```
fraud-detection-demo/
├── app.py                      # Gradio interface
├── model.pkl                   # Trained XGBoost model
├── sample_transactions.json    # Preset transaction scenarios
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Adding Your Own Model

If you want to use your own trained model:

### From AWS SageMaker:

```python
import boto3
import pickle

# Download model from S3
s3 = boto3.client('s3')
s3.download_file('your-bucket', 'model/model.tar.gz', 'model.tar.gz')

# Extract and save as pickle
import tarfile
with tarfile.open('model.tar.gz') as tar:
    tar.extractall()

# Save as model.pkl for the demo
with open('model.pkl', 'wb') as f:
    pickle.dump(your_model, f)
```

### From Local Training:

```python
import pickle

# After training your XGBoost model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

## 📈 Future Enhancements

- [ ] Add SHAP explanations for predictions
- [ ] Real-time transaction input (custom amounts)
- [ ] Batch processing for multiple transactions
- [ ] Historical fraud pattern visualization
- [ ] Model performance dashboard

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

## 🙏 Acknowledgments

- Dataset from [Machine Learning Group - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Built with AWS SageMaker and HuggingFace Gradio
- Inspired by real-world fraud detection systems

---

⭐ **Star this repo if you find it helpful!** ⭐
