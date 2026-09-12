# Roboflow YOLOv8 Export

This folder is the target for a Roboflow Universe YOLOv8 dataset export.

Set these environment variables with the project/version you choose from
Roboflow Universe:

```powershell
$env:ROBOFLOW_API_KEY = "your_api_key"
$env:ROBOFLOW_WORKSPACE = "workspace_name"
$env:ROBOFLOW_PROJECT = "project_slug"
$env:ROBOFLOW_VERSION = "1"
```

Then run:

```powershell
py -3.13 scripts\download_roboflow_yolo.py
py -3.13 scripts\merge_yolo_sources.py
```

After merging, training uses:

```text
data/plate-dataset/datacluster-roboflow-yolo/data.yaml
```
