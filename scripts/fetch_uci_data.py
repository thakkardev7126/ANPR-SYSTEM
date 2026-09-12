"""Fetch UCI datasets and produce an OCR-character confusion report.

Examples:
  python scripts/fetch_uci_data.py --dataset letters --output data/uci
  python scripts/fetch_uci_data.py --dataset vehicles --output data/uci
"""
import argparse
import json
import pathlib


DATASETS = {
    "letters": {"id": 59, "target": "letter"},
    "vehicles": {"id": 149, "target": "class"},
}


def load_dataset(dataset_name):
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit("Install ucimlrepo to fetch UCI data: pip install ucimlrepo") from exc
    return fetch_ucirepo(id=DATASETS[dataset_name]["id"])


def run_letter_confusion(dataset, output_dir):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit("Install scikit-learn to build the confusion report") from exc

    features = dataset.data.features
    labels = dataset.data.targets.iloc[:, 0].astype(str)
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(train_x, train_y)
    predicted = classifier.predict(test_x)
    alphabet = sorted(labels.unique())
    matrix = confusion_matrix(test_y, predicted, labels=alphabet)
    rows = []
    for row_index, character in enumerate(alphabet):
        confusions = []
        for column_index, other in enumerate(alphabet):
            if character != other and matrix[row_index, column_index]:
                confusions.append({"character": other, "count": int(matrix[row_index, column_index])})
        rows.append({"character": character, "confusions": confusions})

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": "UCI Letter Recognition",
        "purpose": "Character similarity evidence for bounded ANPR OCR correction",
        "labels": alphabet,
        "accuracy": float((predicted == test_y).mean()),
        "confusions": rows,
        "warning": "Do not apply these substitutions without Indian plate format validation.",
    }
    destination = output_dir / "letter_confusion.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="letters")
    parser.add_argument("--output", default="data/uci")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output)
    dataset = load_dataset(args.dataset)
    if args.dataset == "letters":
        destination = run_letter_confusion(dataset, output_dir)
        print(f"Wrote {destination}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        features = dataset.data.features
        targets = dataset.data.targets
        features.to_csv(output_dir / "vehicle_features.csv", index=False)
        targets.to_csv(output_dir / "vehicle_targets.csv", index=False)
        print(f"Wrote vehicle data to {output_dir}")


if __name__ == "__main__":
    main()
