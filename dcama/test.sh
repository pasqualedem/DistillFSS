python ./test_dcama.py --datapath "./data/Pascal/VOCdevkit" \
                 --benchmark pascal \
                 --fold 0 \
                 --bsz 1 \
                 --nworker 8 \
                 --backbone swin \
                 --feature_extractor_path "./checkpoints/swin_base_patch4_window12_384.pth" \
                 --logpath "./logs" \
                 --load "./checkpoints/dcama/pascal/fold0_dcama_swin_xm0xgeoo.safetensors" \
                 --nshot 1 \
                 --vispath "./vis_1" \
                #  --visualize
