"""
Downloads portrait images per prakriti class using icrawler.
Install: pip install icrawler
Run:     python download_dataset.py
"""

import os

try:
    from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
except ImportError:
    print("icrawler not installed. Run: pip install icrawler")
    exit(1)

DATASET_DIR     = "dataset"
IMAGES_PER_QUERY = 40   # multiple queries per class → more variety

# Multiple search queries per class for diversity
SEARCH_QUERIES = {
    "vata": [
        "thin oval face angular jawline woman portrait",
        "slim face narrow chin woman close up",
        "elongated face thin features woman headshot",
        "vata body type thin face woman",
    ],
    "pitta": [
        "heart shaped face sharp features woman portrait",
        "pointed chin sharp cheekbones woman headshot",
        "triangular face strong jawline woman portrait",
        "medium build sharp face woman close up",
    ],
    "kapha": [
        "round full face soft features woman portrait",
        "chubby round face woman headshot",
        "wide face soft jawline woman portrait",
        "full cheeks round face woman close up",
    ],
    "vata_pitta": [
        "oval face sharp cheekbones woman portrait",
        "angular oval face woman headshot",
        "slim face defined cheekbones woman portrait",
        "narrow face sharp features woman close up",
    ],
    "vata_kapha": [
        "oval round face soft jawline woman portrait",
        "slightly round oval face woman headshot",
        "soft oval face woman portrait",
        "medium oval face gentle features woman",
    ],
    "pitta_kapha": [
        "heart round face soft features woman portrait",
        "wide heart shaped face woman headshot",
        "round face with defined features woman portrait",
        "full face medium features woman close up",
    ],
}

for class_name, queries in SEARCH_QUERIES.items():
    save_dir = os.path.join(DATASET_DIR, class_name)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([
        f for f in os.listdir(save_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))
    ])
    print(f"\n[{class_name}] Existing: {existing} images")

    for q_idx, query in enumerate(queries):
        print(f"  Downloading: \"{query}\"")
        crawler = BingImageCrawler(
            storage={"root_dir": save_dir},
            feeder_threads=2,
            parser_threads=2,
            downloader_threads=4,
        )
        crawler.crawl(
            keyword=query,
            max_num=IMAGES_PER_QUERY,
            min_size=(150, 150),
            file_idx_offset='auto'
        )

    new_count = len([
        f for f in os.listdir(save_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))
    ])
    print(f"  ✅ {class_name}: {existing} → {new_count} images")

print("\n✅ Download complete! Run: python augment_dataset.py")
