import gradio as gr
import pickle
import numpy as np
import json

# Load the trained XGBoost model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model file not found. Please add model.pkl to the repository.")
    model = None

# Load sample transactions
with open('sample_transactions.json', 'r') as f:
    samples = json.load(f)

def predict_fraud(scenario_name):
    """Predict fraud for selected scenario"""
    if model is None:
        return "❌ Model not loaded. Please check model.pkl file.", 0.0, {}

    # Get transaction features from selected scenario
    scenario = samples[scenario_name]
    features = np.array(scenario['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    # Format result
    result = "🚨 FRAUDULENT" if prediction == 1 else "✅ LEGITIMATE"
    fraud_prob = probability[1] * 100
    legit_prob = probability[0] * 100

    # Transaction details
    details = {
        "Amount": f"${scenario['amount']:.2f}",
        "Description": scenario['description'],
        "Risk Level": get_risk_level(fraud_prob)
    }

    return result, fraud_prob, details

def get_risk_level(prob):
    """Get risk level based on fraud probability"""
    if prob < 20:
        return "🟢 Low Risk"
    elif prob < 50:
        return "🟡 Medium Risk"
    elif prob < 80:
        return "🟠 High Risk"
    else:
        return "🔴 Critical Risk"

# Create Gradio interface
with gr.Blocks(title="Credit Card Fraud Detection") as demo:
    gr.Markdown("""
    # 💳 Credit Card Fraud Detection System

    **Production ML System** - XGBoost model trained on 284K transactions

    Select a transaction scenario below to see real-time fraud detection in action!
    """)

    with gr.Row():
        with gr.Column():
            scenario_dropdown = gr.Dropdown(
                choices=list(samples.keys()),
                value=list(samples.keys())[0],
                label="📋 Select Transaction Scenario",
                info="Choose from realistic transaction patterns"
            )

            predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")

            gr.Markdown("""
            ### About This Demo
            - **Model**: XGBoost trained on 284,807 transactions
            - **Dataset**: Kaggle Credit Card Fraud Dataset (PCA-transformed features)
            - **Accuracy**: 99.9% on test set
            - **Backend**: AWS SageMaker training pipeline
            """)

        with gr.Column():
            result_text = gr.Textbox(label="🎯 Prediction Result", interactive=False, text_align="center")
            fraud_probability = gr.Number(label="⚠️ Fraud Probability (%)", precision=2)

            transaction_details = gr.JSON(label="📊 Transaction Details")

            gr.Markdown("""
            ### Risk Levels
            - 🟢 **Low Risk** (0-20%): Normal transaction
            - 🟡 **Medium Risk** (20-50%): Review recommended
            - 🟠 **High Risk** (50-80%): Suspicious activity
            - 🔴 **Critical Risk** (80-100%): Likely fraud
            """)

    predict_btn.click(
        fn=predict_fraud,
        inputs=[scenario_dropdown],
        outputs=[result_text, fraud_probability, transaction_details]
    )

    gr.Markdown("""
    ---
    ### 🔗 Links
    - **GitHub**: [fraud-detection-mlops](https://github.com/Donald8585/fraud-detection-mlops) (AWS Production Pipeline)
    - **Portfolio**: [Donald8585](https://github.com/Donald8585/)

    Built with ❤️ using XGBoost, Gradio, and AWS SageMaker
    """)

if __name__ == "__main__":
    demo.launch()
