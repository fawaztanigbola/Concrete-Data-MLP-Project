# Concrete Compressive Strength Prediction — MLP

A PyTorch multi-layer perceptron (MLP) that predicts concrete compressive strength from mixture composition and age, trained on the UCI Concrete Compressive Strength dataset.

## Overview

Concrete strength depends on several ingredients (cement, water, aggregates, etc.) and curing age. This project builds a small feedforward neural network to learn that relationship as a regression problem.

## Dataset

- **Source:** [UCI Machine Learning Repository — Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- **File:** `Concrete_Data.csv`
- **Samples:** 1030
- **Features (8):** cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, age
- **Target (1):** concrete compressive strength (MPa)

> Place `Concrete_Data.csv` in the project root before running the script.

## Model Architecture

A simple deep MLP (`ConcreteMLP`):

```
Input (8) → Linear(8, 16) → ReLU
          → Linear(16, 1)
```

## Pipeline

1. **Load data** with pandas and split into features (`X`) and target (`y`).
2. **Standardize features** using `sklearn.preprocessing.StandardScaler`.
3. **Convert to tensors** (`float32`).
4. **Shuffle and split** into train/test sets (80/20) using `torch.randperm` with a fixed seed (`42`) for reproducibility.
5. **Train** for 300 epochs using:
   - Loss: `MSELoss`
   - Optimizer: `Adam` (`lr=0.01`)
6. **Log** training loss every 50 epochs.

## Requirements

```
pandas
torch
scikit-learn
```

Install with:

```bash
pip install pandas torch scikit-learn
```

## Usage

```bash
python concrete_data_mlp.py
```

Expected output (loss values will vary slightly by run/environment):

```
Epoch [50/300], Loss: ...
Epoch [100/300], Loss: ...
...
Epoch [300/300], Loss: ...
```

## Project Structure

```
Concrete_data_MLP/
├── Concrete_Data.csv
├── concrete_data_mlp.py
└── README.md
```

## Notes / Next Steps

- Test set (`x_test`, `y_test`) is currently prepared but not yet evaluated — add a periodic test-loss check to monitor overfitting.
- Consider adding early stopping, learning rate scheduling, or a validation split for more robust training.
- Consider plotting the training loss curve to visualize convergence.

## License

This project is for educational purposes as part of a deep learning course.
