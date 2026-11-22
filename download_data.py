# -*- coding: utf-8 -*-
"""Download COCO128 dataset and YOLOv8n weights.

Two assets on GitHub releases:
  - v0.0.0 : coco128_with_yaml.zip (dataset)
  - v8.2.0 : yolov8n.pt (weights)

Direct HTTPS to github.com may time out on some networks; a `gh release download`
fallback is used when urllib fails.
"""
import urllib.request, os, zipfile, subprocess, sys

BASE = "https://github.com/ultralytics/assets/releases/download"
DATA = os.path.join("data")
os.makedirs(DATA, exist_ok=True)

def dl(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print("exists, skip:", dest); return
    print("downloading", url)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print("urllib failed:", e, "-> trying gh release download")
        # parse tag + pattern from url: .../download/<tag>/<pattern>
        parts = url.split("/download/")
        tag, pattern = parts[1].split("/", 1)
        subprocess.run(["gh", "release", "download", tag, "-R", "ultralytics/assets",
                        "-p", pattern, "--clobber"], check=True)
        # gh writes to cwd; move to dest
        if os.path.basename(pattern) != os.path.basename(dest):
            os.replace(os.path.basename(pattern), dest)
    print("ok", dest, os.path.getsize(dest))

zip_path = os.path.join(DATA, "coco128_with_yaml.zip")
dl(BASE + "/v0.0.0/coco128_with_yaml.zip", zip_path)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(DATA)
print("extracted coco128_with_yaml")

dl(BASE + "/v8.2.0/yolov8n.pt", os.path.join(DATA, "yolov8n.pt"))
print("done")