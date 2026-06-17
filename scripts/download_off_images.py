#!/usr/bin/env python3
"""Download product images with ingredient lists from Open Food Facts.

Saves images to `data/ocr_test_images/` and a CSV `labels.csv` mapping filenames to ingredient text.
"""
import requests
import os
import csv
import argparse
from time import sleep


API_URL = "https://world.openfoodfacts.org/cgi/search.pl"


def fetch_products(page, page_size=100):
    params = {
        'action': 'process',
        'page': page,
        'page_size': page_size,
        'json': 1,
        'fields': 'code,product_name,ingredients_text,ingredients_text_en,image_ingredients_url,image_front_url,image_url'
    }
    headers = {'User-Agent': 'ENHANCING-FoodLabelBot/1.0 (+https://example.org)'}
    r = requests.get(API_URL, params=params, timeout=30, headers=headers)
    r.raise_for_status()
    return r.json()


def download_image(url, out_path):
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(1024*8):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def main(dest_dir='data/ocr_test_images', desired=30, page_size=100):
    os.makedirs(dest_dir, exist_ok=True)
    labels_path = os.path.join(dest_dir, 'labels.csv')

    collected = 0
    page = 1
    rows = []

    print(f"Collecting up to {desired} images from Open Food Facts...")
    while collected < desired:
        try:
            data = fetch_products(page, page_size=page_size)
        except Exception as e:
            print(f"Warning: failed to fetch page {page}: {e}")
            page += 1
            sleep(1)
            continue
        products = data.get('products', [])
        if not products:
            break

        for p in products:
            if collected >= desired:
                break

            ingredients = p.get('ingredients_text') or p.get('ingredients_text_en')
            img = p.get('image_ingredients_url') or p.get('image_front_url') or p.get('image_url')
            code = p.get('code') or f'p{page}_{collected}'

            if not ingredients or not img:
                continue

            ext = os.path.splitext(img)[1].split('?')[0]
            if ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
                ext = '.jpg'

            fname = f"{code}{ext}"
            out_path = os.path.join(dest_dir, fname)

            if download_image(img, out_path):
                rows.append((fname, ingredients))
                collected += 1
                print(f"Downloaded {collected}: {fname}")
            else:
                print(f"Failed to download image for product {code}")

        page += 1
        sleep(0.5)

    # write labels
    with open(labels_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'true_labels'])
        for r in rows:
            writer.writerow(r)

    print(f"Done. Saved {collected} images to {dest_dir}")
    print(f"Labels CSV: {labels_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dest', default='data/ocr_test_images')
    p.add_argument('--count', type=int, default=30)
    args = p.parse_args()
    main(dest_dir=args.dest, desired=args.count)
