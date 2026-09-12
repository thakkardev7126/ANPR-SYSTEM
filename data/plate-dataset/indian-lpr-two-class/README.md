# Indian_LPR Two-Class YOLO Dataset

This folder is the generated target for Indian_LPR-style annotations.

The Indian_LPR GitHub repository says the full road dataset is not public due
to legal restrictions, so the project does not auto-download it. After placing
an authorized local copy under `data/plate-dataset/Indian_LPR`, run:

```powershell
py -3.13 scripts\prepare_indian_lpr_two_class.py --source data\plate-dataset\Indian_LPR
```

The converter writes `data.yaml` with:

```yaml
names:
  0: plate_single_line
  1: plate_double_line
```

After `data.yaml` exists here, `backend/train_resume.py` automatically uses it
unless `ANPR_DATA_YAML` points somewhere else.
