#!/usr/bin/env python3
"""Run OCR over downloaded images and save extracted text alongside true ingredient labels."""
import os
import csv
import time
import argparse
import ast
import numpy as np
import pandas as pd
import easyocr
from tqdm import tqdm


def preprocess_text(text):
    if text is None:
        return ""
    return text.lower().strip()


def labels_to_binary(label_list, classes):
    binary = np.zeros((len(label_list), len(classes)))
    for i, labels in enumerate(label_list):
        for l in labels:
            if l in classes:
                binary[i][classes.index(l)] = 1
    return binary


def safe_parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def run(reader, image_dir='data/ocr_test_images', labels_csv=None, out_csv='pipeline_results_ocr_sim.csv'):
    # load labels
    labels = {}
    if labels_csv and os.path.exists(labels_csv):
        with open(labels_csv, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                labels[row['filename']] = row['true_labels']

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
    rows = []

    for img in tqdm(image_files, desc='OCR'):
        path = os.path.join(image_dir, img)
        start = time.time()
        try:
            results = reader.readtext(path, detail=0)
            extracted = ' '.join(results)
        except Exception:
            extracted = ''
        elapsed = time.time() - start

        processed = preprocess_text(extracted)
        ocr_failed = 1 if processed.strip() == "" else 0
        true = labels.get(img, '')

        rows.append({'filename': img, 'text': processed, 'true_labels': true, 'ocr_failed': ocr_failed, 'ocr_time': elapsed})

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved OCR results to {out_csv}")
    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='data/ocr_test_images')
    p.add_argument('--labels', default='data/ocr_test_images/labels.csv')
    p.add_argument('--out', default='pipeline_results_ocr_sim.csv')
    args = p.parse_args()

    reader = easyocr.Reader(['en'], gpu=False)
    run(reader, image_dir=args.dir, labels_csv=args.labels, out_csv=args.out)
