# Concrete Compressive Strength Prediction — MLP

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A PyTorch-based Multi-Layer Perceptron (MLP) designed to predict the compressive strength of concrete based on its ingredient mixture and curing age. This repository implements a complete regression pipeline using the classic UCI Concrete Compressive Strength dataset.

---

## 📖 Overview

Concrete compressive strength is a highly non-linear function of its ingredients (cement, water, aggregates, etc.) and its aging/curing time. This project implements a feedforward neural network (MLP) in PyTorch to model these complex relationships, providing an automated way to estimate concrete strength without waiting for physical 28-day compression tests.

---

## 🛠️ Tech Stack

| Technology / Library | Purpose |
| :--- | :--- |
| **Python** | Core programming language |
| **PyTorch** | Deep learning framework for model architecture and training |
| **Pandas** | Data loading and manipulation |
| **Scikit-Learn** | Feature scaling and preprocessing (`StandardScaler`) |
| **Matplotlib** | Data visualization and loss curve plotting |

---

## 📊 Dataset Details

- **Source:** [UCI Machine Learning Repository — Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- **File Name:** `Concrete_Data.csv`
- **Dataset Size:** 1,030 samples
- **Input Features (8):**
  1. Cement ($kg/m^3$)
  2. Blast Furnace Slag ($kg/m^3$)
  3. Fly Ash ($kg/m^3$)
  4. Water ($kg/m^3$)
  5. Superplasticizer ($kg/m^3$)
  6. Coarse Aggregate ($kg/m^3$)
  7. Fine Aggregate ($kg/m^3$)
  8. Age (days, 1~365)
- **Target Variable (1):** Concrete Compressive Strength (MPa)

> ⚠️ **Note:** Ensure `Concrete_Data.csv` is placed in the project root directory before executing the training script.

---

## 🧠 Model Architecture

The network (`ConcreteMLP`) is a lightweight, fully-connected feedforward neural network optimized for tabular regression:

```
       Input Features (8)
               │
        ┌──────▼──────┐
        │ Linear(8,16)│
        └──────┬──────┘
               │
            [ReLU]
               │
        ┌──────▼──────┐
        │ Linear(16,1)│
        └──────┬──────┘
               │
               ▼
     Predicted Strength (1)
```

---

## 🔄 Pipeline Workflow

1. **Data Ingestion:** Loads `Concrete_Data.csv` using `pandas` and separates features ($X$) from the target ($y$).
2. **Feature Scaling:** Standardizes features using `sklearn.preprocessing.StandardScaler` to ensure stable gradient descent.
3. **Tensor Conversion:** Converts data arrays into PyTorch `float32` tensors.
4. **Data Splitting:** Shuffles and splits the dataset into training (80%) and testing (20%) subsets using `torch.randperm` with a fixed seed (`42`) for reproducibility.
5. **Model Training:** Trains the network for 300 epochs using:
   - **Loss Function:** Mean Squared Error (`MSELoss`)
   - **Optimizer:** Adam (`learning_rate = 0.01`)
6. **Logging:** Outputs training loss progress every 50 epochs.

---

## 🚀 Getting Started & Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fawaztanigbola/Concrete-Data-MLP-Project.git
   cd Concrete-Data-MLP-Project
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If `scikit-learn` is not automatically installed, install it manually via:*
   ```bash
   pip install scikit-learn
   ```

---

## 💻 Usage

To run the training pipeline, execute the main script:

```bash
python concrete_data_mlp.py
```

### Expected Output

During execution, you should see training logs similar to the following (exact loss values may vary slightly depending on your environment):

```text
Epoch [50/300], Loss: 245.1234
Epoch [100/300], Loss: 112.4567
Epoch [150/300], Loss: 85.3210
Epoch [200/300], Loss: 72.1456
Epoch [250/300], Loss: 65.9876
Epoch [300/300], Loss: 61.2345
```

---

## 📂 Project Structure

```text
Concrete-Data-MLP-Project/
├── Concrete_Data.csv        # Dataset file (must be present in root)
├── concrete_data_mlp.py     # Main PyTorch training script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🎯 Notes & Next Steps

- **Test Set Evaluation:** The test set (`x_test`, `y_test`) is currently prepared but not evaluated during the training loop. Adding a periodic validation/test loss check will help monitor and prevent overfitting.
- **Regularization:** Consider adding Dropout layers or $L_2$ regularization (weight decay in the Adam optimizer) if overfitting occurs.
- **Optimization:** Implement early stopping, learning rate scheduling, or a validation split for more robust training.
- **Visualization:** Use `matplotlib` to plot the training and validation loss curves to visually inspect convergence.

---

## 📄 License

This project is open-source and intended for educational purposes as part of a deep learning curriculum. Feel free to use, modify, and distribute.