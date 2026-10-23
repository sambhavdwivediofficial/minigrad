# MiniGrad: Setup, Training, Prediction, and Evaluation

## 1. Setup and Initial Training

### Check Python Version

```powershell
python --version
```

### Install Project Dependencies

```powershell
python -m pip install -e ".[dev]"
```

### Install With Optional Plotting Support

```powershell
python -m pip install -e ".[dev,viz]"
```

### Run Tests

```powershell
python -m pytest -q
```

### Train the Model

```powershell
python .\examples\train_xor.py
```

## 2. Model Prediction

### Predict Class 1

```powershell
python .\examples\predict_xor.py 0.8 0.4
```

### Predict Class 0

```powershell
python .\examples\predict_xor.py -0.8 0.4
```

## 3. Model Evaluation

### Evaluate the Model

```powershell
python .\examples\evaluate_xor.py
```

### Evaluate on 5,000 Samples

```powershell
python .\examples\evaluate_xor.py --samples 5000 --seed 42
```

### Change the Classification Threshold

```powershell
python .\examples\evaluate_xor.py --threshold 0.6
```

**Note:** Evaluation requires training to be completed and `artifacts/xor_model.npz` to be available.

## 4. Customize Model Training

### Increase Training Epochs

```powershell
python .\examples\train_xor.py --epochs 1500
```

### Change the Learning Rate

```powershell
python .\examples\train_xor.py --lr 0.02
```

### Increase Early Stopping Patience

```powershell
python .\examples\train_xor.py --patience 150
```

### Change the Random Seed

```powershell
python .\examples\train_xor.py --seed 12
```

### Save a Separate Model Checkpoint

```powershell
python .\examples\train_xor.py --checkpoint artifacts\custom_xor.npz
```

## 5. Use a Custom Checkpoint

### Predict Using the Custom Model

```powershell
python .\examples\predict_xor.py 0.8 0.4 --checkpoint artifacts\custom_xor.npz
```

### Evaluate the Custom Model

```powershell
python .\examples\evaluate_xor.py --checkpoint artifacts\custom_xor.npz
```

These commands let you compare different training configurations and model checkpoints.

## 6. Regression Example

### Train the Regression Model

```powershell
python .\examples\regression_demo.py
```

### Train and Save a Loss-Curve Plot

```powershell
python .\examples\regression_demo.py --plot artifacts\regression_loss.png
```

### Change Loss Sensitivity (Huber Delta)

```powershell
python .\examples\regression_demo.py --delta 0.5
```

## 7. Two-Moons and Spiral Classification

### Run Both Tasks With Plots

```powershell
python .\examples\moons_spiral_classification.py --plot
```

### Run Only the Two-Moons Task

```powershell
python .\examples\moons_spiral_classification.py --task moons --plot
```

### Run Only the Spiral Task With 4 Classes

```powershell
python .\examples\moons_spiral_classification.py --task spiral --spiral-classes 4 --plot
```

**Note:** `--plot` requires `pip install -e ".[viz]"` (matplotlib) and writes PNGs under `artifacts/`.

## 8. Tabular CSV Example

### Generate a Synthetic Housing CSV and Train On It

```powershell
python .\examples\tabular_csv_demo.py
```

### Use a Larger Synthetic Dataset

```powershell
python .\examples\tabular_csv_demo.py --size 5000
```

This example demonstrates the real-tabular-data path: `CSVDataset` loads the file, the script standardizes features/target, and a `Linear + BatchNorm1d + ReLU` network trains with `AdamW`.
