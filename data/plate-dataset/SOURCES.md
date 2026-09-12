# ANPR Dataset Sources

## Active Training Dataset

`backend/train_resume.py` currently selects:

```text
C:/Users/Dev/Downloads/Model/data/plate-dataset/anpr-augmented-yolo/data.yaml
```

This dataset was generated from the merged labelled YOLO sources and then expanded with Albumentations fog, rain, mud, blur, and folded/perspective variants.

Current local counts:

| Split | Images | Labels |
| --- | ---: | ---: |
| train | 9907 | 9907 |
| val | 416 | 416 |

Current class layout:

```yaml
names:
  0: license_plate
```

## Integrated Sources

| Source | Local status | Notes |
| --- | --- | --- |
| Kaggle `kedarsai/indian-license-plates-with-labels` | Integrated as `data/plate-dataset/kaggle-yolo` | Baseline labelled YOLO source. |
| Hugging Face `Dataclusterlabspvtltd/indian-number-plates-dataset` | Integrated as `data/plate-dataset/datacluster-yolo` | Public DataCluster sample, converted from Pascal VOC XML to YOLOv8. |
| Roboflow Universe Indian License Plate datasets | Downloader ready at `scripts/download_roboflow_yolo.py` | Requires your Roboflow API key, workspace, project slug, and version. |

## Reference Sources Not Fully Downloadable Here

| Source | Why it is not directly merged yet |
| --- | --- |
| DataCluster `Indian-Number-Plates-Dataset` GitHub | Public repo points to the Kaggle sample and full dataset contact route. Full YOLO/COCO/Pascal export may require DataCluster access. |
| DataCluster `Indian-Licence-Plate-Image-Dataset` GitHub | Public repo exposes sample images, but the train-ready labelled export is not in the public GitHub tree. |
| Indian_LPR / Indian Licence Plate Dataset in the Wild | Public repo says the full road dataset is not public due to legal restrictions. The converter is ready at `scripts/prepare_indian_lpr_two_class.py` once an authorized copy is placed locally. |

## Rebuild Commands

```powershell
py -3.13 scripts\prepare_kaggle_yolo.py
py -3.13 scripts\prepare_datacluster_yolo.py

$env:ROBOFLOW_API_KEY = "your_api_key"
$env:ROBOFLOW_WORKSPACE = "workspace_name"
$env:ROBOFLOW_PROJECT = "project_slug"
$env:ROBOFLOW_VERSION = "1"
py -3.13 scripts\download_roboflow_yolo.py

py -3.13 scripts\merge_yolo_sources.py
py -3.13 scripts\augment_yolo_albumentations.py --per-kind 1
py -3.13 backend\train_resume.py --dry-run
py -3.13 backend\train_resume.py
```

Use `ANPR_DATA_YAML` to force a different dataset:

```powershell
$env:ANPR_DATA_YAML = "C:\Users\Dev\Downloads\Model\data\plate-dataset\datacluster-roboflow-yolo\data.yaml"
```

