def generate_model_selection_prompt_from_timeseries(name, num_signals):

    user_message = f"""
You are an expert in model selection for anomaly detection on time series data.

## Task:
- Given the information of a dataset and a set of models, select the model you believe will achieve the best performance for detecting anomalies in this dataset. Provide a brief explanation of your choice.

## Dataset Information:
- Dataset Name: {name}
- Number of Signals: {num_signals}

## Model Options:
Autoformer: A Transformer-based model that captures long-term dependencies in time series data through an auto-correlation mechanism.

DLinear: A model that decomposes time series into trend and seasonal components, applying linear transformations for efficient forecasting.

ETSformer: Integrates exponential smoothing techniques into the Transformer architecture to enhance time series forecasting accuracy.

FEDformer: Combines frequency domain analysis with Transformer models to improve long-term time series forecasting performance.

Informer: Utilizes a ProbSparse self-attention mechanism to efficiently handle long sequence time series forecasting tasks.

LightTS: A lightweight model employing sampling-oriented MLP structures for fast and efficient multivariate time series forecasting.

Pyraformer: Introduces a pyramidal attention mechanism to capture long-range dependencies in time series data with reduced complexity.

Reformer: An efficient Transformer variant that uses locality-sensitive hashing for scalable attention in long sequences.

TimesNet: Transforms 1D time series into 2D representations to model temporal variations using convolutional neural networks.

Transformer: A deep learning model leveraging self-attention mechanisms to capture dependencies in sequential data, widely used in time series analysis.


## Rules:
1. Availabel options include Autoformer	DLinear	ETSformer	FEDformer	Informer	LightTS	Pyraformer	Reformer	TimesNet	Transformer
2. Treat all models equally and evaluate them based on their compatibility with the dataset information and the anomaly detection task.
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
