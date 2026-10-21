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
