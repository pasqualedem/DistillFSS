python -u -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 --node_rank=0 --master_port=16005 \
./train_dcama.py --datapath "../PANet/data/Pascal/VOCdevkit" \
           --benchmark pascal \
           --fold 0 \
           --bsz 2 \
           --nworker 8 \
           --backbone swin \
           --feature_extractor_path "checkpoints/swin_base_patch4_window12_384.pth" \
           --logpath "./logs" \
           --lr 1e-3 \
           --nepoch 50 \
           --remove_support_skip