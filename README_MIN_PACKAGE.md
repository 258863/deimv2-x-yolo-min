# DEIMv2 X / YOLO-data fine-tuning — minimal package

Only what's needed to fine-tune the X (DINOv3) model on your own YOLO-format
data. Trimmed out: datasets/, outputs/, docs/, figures/, most of tools/,
.git, other model sizes' checkpoints, and the dinov3 backbone pretrain
weight (not needed -- `-t` loads the full X checkpoint, which already
contains the trained backbone). The pretrained COCO checkpoint itself
(`ckpts/deimv2_coco/.../model.pth`) is also not committed to this repo --
see `.gitignore` for the expected local path.

`classification/` is included: it's the architecture code for the two
"innovation" modules (DEG axis fusion + STA cross-branch exchange) also
referenced by `configs/classification/*.yml`, and `engine/backbone/dinov3_adapter.py`
now lets the detection backbone opt into the same two modules (see
`configs/deimv2/deimv2_dinov3_x_yolo_innovations.yml` below) -- that wiring
has not been trained/evaluated for detection yet, only for classification.

## Setup on the new machine
1. `pip install -r requirements.txt` (plus torch/torchvision matching your CUDA).
2. Point `configs/deimv2/deimv2_dinov3_x_yolo.yml`'s `yolo_data:` at your
   `data.yaml` (or pass `-u yolo_data=/abs/path/to/data.yaml` on the command line).
3. Run:
   python train.py -c configs/deimv2/deimv2_dinov3_x_yolo.yml \
     -t ckpts/deimv2_coco/DEIMv2_DINOv3_X_COCO/model.pth \
     --use-amp --seed 42 -d cuda \
     -u yolo_data=/absolute/path/to/dataset/data.yaml
4. Optional sanity check: python -m unittest tests/test_letterbox.py -v
5. Optional, experimental: swap in `-c configs/deimv2/deimv2_dinov3_x_yolo_innovations.yml`
   to train with the DEG axis-fusion + STA cross-branch exchange modules enabled
   in the detection backbone (same `-t`/checkpoint still applies; the new modules'
   own weights start from scratch since they're not in the COCO checkpoint).
