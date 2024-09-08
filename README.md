
## Dataset Paths
Update paths to your local dataset in `train.py` using `os.path`:
- Image directories:
  - Normal: `~/Downloads/chest_xray/train/NORMAL`
  - Pneumonia: `~/Downloads/chest_xray/train/PNEUMONIA`
- CSV files:
  - Normal: `~/Downloads/synthesized_dataset/train/Normal.csv`
  - Pneumonia: `~/Downloads/synthesized_dataset/train/Pneumonia.csv`

## Metrics
The training loop tracks the following metrics:
- **Training/Test Loss**
- **Accuracy**
- **Precision**
- **Recall**

Run the training using:
```bash
python train.py
