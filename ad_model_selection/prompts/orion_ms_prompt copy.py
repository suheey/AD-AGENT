def generate_model_selection_prompt(name):

    user_message = f"""
You are an expert in model selection for anomaly detection on time series data.

## Task:
- Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}

## Model Options:
- Auto-Encoder with Regression for Time Series Anomaly Detection (AER)
- Time Series Anomaly Detection with Association Discrepancy (Anomaly Transformer)
- Autoregressive Integrated Moving Average (ARIMA)
- Autoencoder Using Dense Layers (Dense AE)
- Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series (GANF)
- Long Short-Term Memory Networks (LSTM)
- Autoencoder using LSTM layers (LSTM AE)
- Time Series Anomaly Detection Using Generative Adversarial Networks (TadGAN)
- A Decoder-only Foundation Model for Time-series Forecasting (TimesFM)
- A Unified Multi-Task Time Series Model (UniTS)
- Variational AutoEncoder (VAE)

## Rules:
1. Availabel options include "AER", "Anomaly Transformer", "ARIMA", "Dense AE", "GANF", "LSTM", "LSTM AE", "TadGAN", "TimesFM", "UniTS", and "VAE."
2. Treat all models equally and evaluate them based on their compatibility with the dataset characteristics and the anomaly detection task.
3. Response Format:
    - Provide responses in a strict **JSON** format with the keys "reason" and "choice."
        - "reason": Your explanation of the reasoning.
        - "choice": The model you have selected for anomaly detection in this dataset.

Response in JSON format:
"""
    
    messages = [
        # {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        # {"role": "assistant", "content": assistant_message}
    ]

    return messages
