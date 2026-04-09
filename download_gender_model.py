"""
Downloads GenderNet Caffe model files.
Run: python download_gender_model.py
"""
import urllib.request
import os
import sys

os.makedirs("age_model", exist_ok=True)

# Multiple mirror sources per file — tries each until one works
FILES = {
    "age_model/gender_deploy.prototxt": [
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt",
    ],
    "age_model/gender_net.caffemodel": [
        "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
    ],
}

def download(path, urls):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print(f"  Already exists: {path}")
        return True
    for url in urls:
        try:
            print(f"  Trying: {url}")
            urllib.request.urlretrieve(url, path)
            size = os.path.getsize(path)
            if size > 1000:
                print(f"  Saved: {path} ({size // 1024} KB)")
                return True
            else:
                os.remove(path)  # incomplete file
        except Exception as e:
            print(f"  Failed: {e}")
    return False

all_ok = True
for path, urls in FILES.items():
    print(f"\nDownloading {path}...")
    if not download(path, urls):
        print(f"  ERROR: Could not download {path} from any source.")
        all_ok = False

if all_ok:
    print("\n✅ Gender model ready. Run: python real_time.py")
else:
    print("\n⚠️  Some files failed. Manual download instructions:")
    print("  1. Go to: https://github.com/smahesh29/Gender-and-Age-Detection")
    print("  2. Download gender_deploy.prototxt and gender_net.caffemodel")
    print("  3. Place both in the age_model/ folder")
