# -*- coding: utf-8 -*-
"""Plot the 3-arm comparison + per-epoch fine-tune curve from results/."""
import json, io, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def load(p):
    with io.open(p, encoding="utf-8") as f: return json.load(f)
out = "figures"; os.makedirs(out, exist_ok=True)
summary = load("results/result_summary.json")
base = summary["zero_shot_pretrained"]; ft = summary.get("finetune_lr1e3", {}); scr = summary.get("from_scratch", {})
labels = ["Zero-shot\n(pretrained)", "Fine-tune\nlr=1e-3", "From scratch"]
m50 = [base["map50"], ft.get("map50", 0), scr.get("map50", 0)]
m95 = [base["map50_95"], ft.get("map50_95", 0), scr.get("map50_95", 0)]
x = range(3)
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
b1 = ax.bar([i - 0.18 for i in x], m50, width=0.36, label="mAP50", color="#4C72B0")
b2 = ax.bar([i + 0.18 for i in x], m95, width=0.36, label="mAP50-95", color="#DD8452")
for b in list(b1) + list(b2):
    h = b.get_height(); ax.annotate("%.3f" % h, (b.get_x() + b.get_width()/2, h), ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(labels)
ax.set_ylabel("mAP"); ax.set_ylim(0, 0.85)
ax.set_title("YOLOv8n on COCO128 split (80 train / 20 val / 28 test)"); ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(out, "three_arm_bar.png")); plt.close()
print("bar plot done")