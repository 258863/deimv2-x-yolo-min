# DEIMv2 X / YOLO-data fine-tuning — minimal package

Only what's needed to fine-tune the X (DINOv3) model on your own YOLO-format
data. Trimmed out: datasets/, outputs/, classification/, docs/, figures/,
tools/, .git, other model sizes' checkpoints, and the dinov3 backbone
pretrain weight (not needed -- `-t` loads the full X checkpoint, which
already contains the trained backbone).

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
