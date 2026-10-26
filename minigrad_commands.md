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

## 9. CNN Shape Classification

### Train on the Default 16x16 Synthetic Shapes

```powershell
python .\examples\cnn_shapes_classification.py
```

### Use Larger Images and More Epochs

```powershell
python .\examples\cnn_shapes_classification.py --image-size 24 --epochs 40
```

### Use a Noisier Dataset

```powershell
python .\examples\cnn_shapes_classification.py --noise 0.15
```

This example trains a small `Conv2d -> ReLU -> MaxPool2d` stack to classify procedurally drawn squares, circles, and crosses — the first real image-classification example in the project.

## 10. Character-Level LSTM Language Model

### Train and Generate Text

```powershell
python .\examples\char_lstm_demo.py
```

### Train a Bigger Model for Longer

```powershell
python .\examples\char_lstm_demo.py --hidden-size 128 --epochs 500
```

### Generate a Longer Sample

```powershell
python .\examples\char_lstm_demo.py --generate-length 300
```

This example trains an `Embedding -> LSTM -> Linear` character-level language model with gradient clipping, then generates text autoregressively — one character sampled at a time, fed back in as the next input.

## 11. Predict Scripts (Every Trained Model)

### Regression

```powershell
python .\examples\predict_regression.py 1.5
```

### Two-Moons

```powershell
python .\examples\predict_moons.py 1.0 0.5
```

### Spiral

```powershell
python .\examples\predict_spiral.py 0.3 0.2
```

### Tabular Housing Price

```powershell
python .\examples\predict_tabular.py 3 1800 15 5.0
```

### CNN Shapes (Random Test Shape)

```powershell
python .\examples\predict_cnn_shapes.py
```

### CNN Shapes (Force a Specific Shape)

```powershell
python .\examples\predict_cnn_shapes.py --true-class circle
```

**Note:** each predict script requires its matching example to have been run at least once first, so the checkpoint it loads exists.

## 12. Transformer Text Generation

### Train and Generate Text

```powershell
python .\examples\char_transformer_demo.py
```

### Train a Deeper Model

```powershell
python .\examples\char_transformer_demo.py --num-layers 3 --d-model 64 --epochs 500
```

This is the same next-character-prediction task as `char_lstm_demo.py`, trained through `TransformerLM` (attention) instead of `LSTM` (recurrence).

## 13. Resumable Training

### Train, Save, Resume, and Compare Against an Uninterrupted Run

```powershell
python .\examples\resume_training_demo.py
```

### Use Longer Segments

```powershell
python .\examples\resume_training_demo.py --first-segment-epochs 300 --second-segment-epochs 300
```

This example proves `save_training_state` / `load_training_state` resume training exactly: it trains a model two ways (interrupted-and-resumed vs. uninterrupted) and prints both final losses, which match to within floating-point precision.
