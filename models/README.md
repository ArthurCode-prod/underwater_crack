# Models

`best_patent_model.pth` is the pretrained PatentUNet-TAFE checkpoint used for the reported evaluation.

To retrain a new checkpoint:

```bash
python scripts/train_model.py --data-dir data/deepcrack --output models/patentunet_tafe_retrained.pth
```
