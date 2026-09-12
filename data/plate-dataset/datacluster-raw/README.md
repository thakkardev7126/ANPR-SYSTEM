---
license: cc-by-nc-nd-4.0
task_categories:
  - object-detection
  - image-classification
  - image-to-text
  - visual-question-answering
size_categories:
  - n<1K
tags:
  - indian-number-plates
  - license-plate-detection
  - anpr
  - number-plate-recognition
  - ocr
  - object-detection
  - computer-vision
  - automotive
  - self-driving
  - lmm
  - vqa
pretty_name: Indian Number Plates Dataset (Sample)
---

# Indian Number Plates Dataset — Sample

> ⚠️ **This is a free sample subset for evaluation purposes only.**  
> The full dataset (15,000+ annotated images, multiple formats) is available for commercial licensing.  
> **Contact:** [sales@datacluster.ai](mailto:sales@datacluster.ai) · [datacluster.ai](https://datacluster.ai)

---

## Dataset Summary

This dataset contains real-world images of Indian vehicle number plates, captured on mobile phones across diverse urban and rural environments throughout India. It is well-suited for training license plate detection and recognition models used in Automatic Number Plate Recognition (ANPR) systems, smart traffic monitoring, automated tolling, parking management, law enforcement, and self-driving applications.

It is also **optimized for Generative AI, Visual Question Answering (VQA), Image Classification, and Large Multimodal Model (LMM) development**, providing a strong basis for achieving robust model performance on India-specific automotive tasks.

Scenes cover a wide variety of real-world capture scenarios — moving and stationary vehicles, dense city traffic, highways, parking lots, and rural roads — under varied lighting conditions (day, night, dusk), distances, viewing angles, and weather. The dataset captures the full diversity of Indian number plate styles across different vehicle types, making it especially valuable for ANPR models that need to perform reliably on Indian roads.

## Classes & Annotations

* **Class name:** `number_plate`
* **Attribute:** Each number plate bounding box includes a `number_plate_text` attribute, which contains all the characters present within the plate — making the dataset suitable for both **detection** and **OCR / recognition** tasks.

### Annotation distribution by vehicle type (full dataset)

| Vehicle type | Share |
| --- | --- |
| Two-wheelers (bike / scooter) | 25% |
| Six+ wheelers (truck / bus / construction truck) | 24% |
| Three-wheelers (auto) | 15% |
| Four-wheelers (car / van) | 14% |
| Commercial vehicles (tempo / traveler van / ambulance) | 12% |
| Tractor | 10% |

## Sample vs. Full Dataset

|  | Sample (this repo) | Full Dataset |
| --- | --- | --- |
| Images | ~200 (subset) | 15,000+ |
| Annotated images | ~200 | 10,000+ |
| Bounding boxes | ~200 | 12,000+ |
| Annotation formats | Pascal VOC (XML) — other formats available on request | COCO, YOLO, Pascal VOC, TF-Record |
| `number_plate_text` attribute | ✅ Included | ✅ Included |
| Locations covered | Representative subset | 700+ cities and villages across India |
| Resolution | HD (1920×1080 and above) | HD (1920×1080 and above) |
| Scene diversity | Representative subset | Full range (urban, rural, day, night, close, far) |
| Commercial use | ❌ Not permitted | ✅ With license |
| Redistribution | ❌ Not permitted | Per license terms |
| Updates | One-time | Ongoing |

**To license the full dataset:** [sales@datacluster.ai](mailto:sales@datacluster.ai)

## Dataset Structure

```
indian-number-plates-dataset/
├── images/                  # JPG images
│   ├── image_0001.jpg
│   └── ...
└── annotations/             # Pascal VOC XML annotations (one per image)
    ├── image_0001.xml
    └── ...
```

Each XML file contains bounding-box annotations in the Pascal VOC format around the number plate region, along with the `number_plate_text` attribute capturing the plate's characters. Filenames match their corresponding images.

> **Need a different annotation format?** This sample ships in Pascal VOC (XML) only. YOLO, COCO, and TF-Record versions are available on request — see the conversion snippet below, or contact [sales@datacluster.ai](mailto:sales@datacluster.ai).

## Data Collection

* **Source:** Real-world mobile phone captures, crowdsourced from 4,000+ contributors
* **Locations:** 700+ cities and villages across urban and rural India
* **Capture period:** 2020–2022
* **Resolution:** 100% HD and above (1920×1080+)
* **Conditions:** Indoor and outdoor scenes; varied lighting (day, night, dusk), weather, distances, and viewing angles
* **Quality:** All images are exclusively owned by DataCluster Labs (not scraped from the internet) and each image is manually reviewed and verified by computer vision professionals at DC Labs
* **Use cases:** Number plate detection, Automatic Number Plate Recognition (ANPR), license plate OCR, smart traffic systems, automated tolling, parking management, self-driving systems, VQA on automotive scenes, and Large Multimodal Model (LMM) training

## How to Use

### Download

```bash
# Using the Hugging Face CLI
huggingface-cli download Dataclusterlabspvtltd/indian-number-plates-dataset --repo-type dataset --local-dir ./indian-number-plates-dataset
```

Or clone directly:

```bash
git lfs install
git clone https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset
```

### Convert VOC to YOLO or COCO

The sample ships in Pascal VOC format. Convert easily with `pylabel`:

```python
from pylabel import importer

# VOC → YOLO
dataset = importer.ImportVOC(path="annotations")
dataset.export.ExportToYoloV5(output_path="annotations_yolo")

# VOC → COCO
dataset.export.ExportToCoco(output_path="annotations_coco/annotations.json")
```

## License

This sample dataset is released under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license.

Key points:

* ✅ Free to download and evaluate
* ✅ Free for academic and non-commercial research with attribution
* ❌ No commercial use without a license from DataCluster Labs
* ❌ No derivative works or modifications for redistribution
* ❌ No use in training commercial ML models

For commercial licensing of the full dataset, contact **[sales@datacluster.ai](mailto:sales@datacluster.ai)**.

## Citation

If you use this dataset in academic work, please cite:

```bibtex
@misc{datacluster_indian_number_plates_sample,
  title        = {Indian Number Plates Dataset (Sample)},
  author       = {DataCluster Labs},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset}},
  note         = {Sample subset. Full dataset available for commercial licensing at sales@datacluster.ai}
}
```

## About DataCluster Labs

DataCluster Labs specializes in managed crowd-sourced data collection and annotation — images, videos, audio, text, and surveys — through our Dailydata platform. We deliver custom datasets for computer vision, NLP, and ML use cases, with a strong focus on India-first data that captures the diversity of real-world conditions across the subcontinent.

📧 **Sales / Full Dataset Access:** [sales@datacluster.ai](mailto:sales@datacluster.ai)  
🌐 **Website:** [datacluster.ai](https://datacluster.ai)