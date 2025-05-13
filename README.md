# OpenAD

**OpenAD** is a multi-agent anomaly detection platform designed to support the full lifecycle of real-world anomaly detection—from data preprocessing and model selection to detection, explanation, and evaluation. It integrates classical and graph-based AD algorithms with LLM-powered modules for enhanced usability, privacy, and adaptability.

> 🔍 One platform. Multiple agents. All your anomaly detection workflows—automated, explainable, and secure.

---

## 🔧 Features

- **Modular Pipeline Execution**: Supports tabular, sequential, and graph anomaly detection algorithms for more and more complex data types.
- **Multi-Agent Architecture**: Detection, explanation, and adaptation are handled by decoupled agents with clear APIs and extendability.
- **LLM Integration** (in progress): Language models assist in tasks such as explanation, synthetic anomaly generation, and interactive debugging.
- **Privacy-Aware Design** (in progress): Includes a framework for anonymizing data before AD processing, suitable for regulated domains.
- **Human-in-the-loop Support** (in progress): Enables analysts to query explanations and iterate on detection results interactively.


---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/USC-FORTIS/OpenAD.git
cd OpenAD
```

### 2. Create and Activate a Virtual Environment

#### On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

#### On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

pip install -r requirements.txt
```

### 4. Set Your OpenAI API Key

Edit the config file to include your OpenAI API key:

```python
# File: /config/config.py

OPENAI_API_KEY = 'your-api-key-here'
```

---

## 🚀 Running the Program

### Run Normally (Sequential Execution)

```bash
python main.py
```

### Run in Parallel Mode

```bash
python main.py -p
```

### Run in Optimizer Mode

```bash
python main.py -o
```

---

## 🧪 Test Commands

You can also run the system with natural-language-like test commands.

### Run a Specific Algorithm

```text
#pyod
Run IForest on ./data/glass_train.mat and ./data/glass_test.mat with contamination=0.1
Run all on ./data/glass_train.mat and ./data/glass_test.mat with contamination=0.1
#pygod
Run DOMINANT on ./data/inj_cora_train.pt and ./data/inj_cora_test.pt
#darts
Run GlobalNaiveAggregate on ./data/yahoo_train.csv and ./data/yahoo_test.csv
#tslib
Run LightTS on ./data/MSL_train.npy and ./data/MSL_test.npy

```

### Run All Algorithms

```text
Run all on ./data/glass_train.mat and ./data/glass_test.mat
```

---

## 📁 Project Structure

```
.
├── config/
│   └── config.py             # Configuration file for API keys
.
.
.
├── data/
│   └── glass.mat             # Sample dataset
├── main.py                   # Main execution script
├── requirements.txt          # Required Python packages
└── README.md                 # Project documentation
```

---

## 📌 Notes

- Make sure your dataset is placed inside the `./data/` directory.
- Modify `main.py` to add support for additional algorithms or datasets if needed.

---

## 👥 Contributors

<a href="https://github.com/USC-FORTIS/OpenAD/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=USC-FORTIS/OpenAD" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

