*Update:*

- Added a test feature to the train.py
For testing purposes, line 14 in train.py can be changed to `development` which will make use of a small dummy data. Switch to `production` to run script on actual data.

- Training script updated to display only the confusion matrix after training is complete.

- Normalized image tensors before fusion.


---
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
```python
python train.py
```