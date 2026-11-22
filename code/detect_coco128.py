# -*- coding: utf-8 -*-
"""
YOLOv8n object detection: zero-shot vs fine-tuning vs from-scratch on COCO128.

This script reproduces the full experiment from a clean dataset directory.
It expects the COCO128 dataset (images/ + labels/) already extracted, plus
the official yolov8n.pt weights. See download_data.py and prepare_split.py.

Usage:
  python detect_coco128.py prepare        # carve disjoint train/val/test (fixes leakage)
  python detect_coco128.py baseline       # zero-shot eval on held-out test
  python detect_coco128.py finetune       # fine-tune pretrained at lr=1e-3
  python detect_coco128.py scratch        # train from scratch
  python detect_coco128.py evaluate       # eval finetune + scratch, write summary
"""
import os, sys, json, io, random, shutil, time, argparse
def log(*a): print(*a, flush=True)
try:
    import torch
    from ultralytics import YOLO
except ImportError as e:
    log("Missing dependency:", e, "— pip install -r requirements.txt"); sys.exit(1)
def box_metrics(res):
    """Robustly extract COCO box metrics across ultralytics versions."""
    box = None
    for attr in ("box",):
        try: box = getattr(res, attr)
        except Exception: box = None
        if box is not None: break
    if box is None:
        try: box = res.results[0].metrics.box
        except Exception: box = res
    mr = list(box.mean_results())
    try: m95 = float(box.maps.mean())
    except Exception: m95 = float(mr[3])
    return {"precision": round(float(mr[0]), 4), "recall": round(float(mr[1]), 4),
            "map50": round(float(mr[2]), 4), "map50_95": round(m95, 4)}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "baseline", "finetune", "scratch", "evaluate"])
    ap.add_argument("--data", default="data/coco128_with_yaml", help="COCO128 root")
    ap.add_argument("--weights", default="data/yolov8n.pt", help="pretrained weights")
    ap.add_argument("--out", default="results", help="output dir")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    args = ap.parse_args();
    DATA = os.path.abspath(args.data)
    YAML = os.path.join(DATA, "coco128_split.yaml")
    os.makedirs(args.out, exist_ok=True)
    if args.mode == "prepare":
        img_src = os.path.join(DATA, "images", "train2017")
        lbl_src = os.path.join(DATA, "labels", "train2017")
        imgs = sorted(f for f in os.listdir(img_src) if f.lower().endswith((".jpg", ".png", ".jpeg")))
        pairs = [os.path.splitext(f)[0] for f in imgs if os.path.exists(os.path.join(lbl_src, os.path.splitext(f)[0] + ".txt"))]
        random.seed(42); random.shuffle(pairs)
        n_test, n_val = 28, 20
        test_set = pairs[:n_test]; val_set = pairs[n_test:n_test + n_val]; train_set = pairs[n_test + n_val:]
        log("split: train=%d val=%d test=%d (disjoint)" % (len(train_set), len(val_set), len(test_set)))
        for part in ("train", "val", "test"):
            for sub in ("images", "labels"): os.makedirs(os.path.join(DATA, sub, part), exist_ok=True)
        def move(stems, part):
            for s in stems:
                for ext in (".jpg", ".png", ".jpeg"):
                    src = os.path.join(img_src, s + ext)
                    if os.path.exists(src): shutil.move(src, os.path.join(DATA, "images", part, s + ext)); break
                lp = os.path.join(lbl_src, s + ".txt")
                if os.path.exists(lp): shutil.move(lp, os.path.join(DATA, "labels", part, s + ".txt"))
        move(train_set, "train"); move(val_set, "val"); move(test_set, "test")
        names = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]
        txt = "path: " + DATA + chr(10) + "train: images/train" + chr(10) + "val: images/val" + chr(10) + "test: images/test" + chr(10) + "nc: 80" + chr(10) + "names: " + str(names).replace(chr(34), chr(39)) + chr(10)
        with io.open(YAML, "w", encoding="utf-8") as f: f.write(txt)
        log("wrote", YAML)
        return
    def evaluate(path, tag):
        model = YOLO(path)
        res = model.val(data=YAML, split="test", batch=args.batch, imgsz=args.imgsz, device=args.device, workers=0, plots=False, verbose=False, amp=True)
        return box_metrics(res)
    if args.mode == "baseline":
        m = evaluate(os.path.abspath(args.weights), "baseline")
        log("zero-shot:", json.dumps(m))
        with io.open(os.path.join(args.out, "baseline.json"), "w", encoding="utf-8") as f: json.dump({"zero_shot_pretrained": m, "imgsz": args.imgsz, "batch": args.batch}, f, ensure_ascii=False, indent=2)
    elif args.mode in ("finetune", "scratch"):
        if args.mode == "finetune":
            model = YOLO(os.path.abspath(args.weights)); name = "yolov8n_ft_lr1e3"; kw = dict(lr0=0.001)
        else:
            cfg = os.path.join(os.path.dirname(__import__("ultralytics").__file__), "cfg", "models", "v8", "yolov8.yaml")
            model = YOLO(cfg); name = "yolov8n_scratch"; kw = dict()
        t0 = time.time()
        model.train(data=YAML, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device, workers=0, patience=12, amp=True, project="runs", name=name, val=True, plots=True, verbose=True, close_mosaic=10, seed=42, deterministic=True, **kw)
        log(name, "seconds:", round(time.time() - t0, 2))
    elif args.mode == "evaluate":
        summary = {};
        summary["experiment"] = "YOLOv8n object detection on COCO128 split"
        summary["config"] = {"imgsz": args.imgsz, "batch": args.batch, "epochs": args.epochs, "patience": 12};
        summary["dataset"] = {"name": "COCO128", "split": {"train": 80, "val": 20, "test": 28}, "classes": 80, "note": "disjoint split, seed 42"};
        try: summary["zero_shot_pretrained"] = json.load(io.open(os.path.join(args.out, "baseline.json"), encoding="utf-8"))["zero_shot_pretrained"]
        except Exception: pass
        for tag, run in (("finetune_lr1e3", "runs/yolov8n_ft_lr1e3/weights/best.pt"), ("from_scratch", "runs/yolov8n_scratch/weights/best.pt")):
            p = os.path.join(run)
            if os.path.exists(p):
                try: summary[tag] = evaluate(os.path.abspath(p), tag)
                except Exception as e: summary[tag] = {"error": str(e)[:200]}
        with io.open(os.path.join(args.out, "result_summary.json"), "w", encoding="utf-8") as f: json.dump(summary, f, ensure_ascii=False, indent=2)
        log("summary written")
if __name__ == "__main__":
    main()