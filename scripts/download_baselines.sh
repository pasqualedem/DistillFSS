CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

download_if_missing () {
    local filename="$1"      # relative path inside CHECKPOINT_DIR
    local file_id="$2"
    local target="$CHECKPOINT_DIR/$filename"
    local tmp_file

    if [ -f "$target" ]; then
        echo "✓ $filename already exists"
        return
    fi

    echo "↓ Downloading $filename"

    # gdown always downloads to CWD; capture actual filename
    tmp_file="$(pwd)/$(basename "$filename")"

    gdown "$file_id" || { echo "Download failed for $filename"; exit 1; }

    if [ ! -f "$tmp_file" ]; then
        echo "Downloaded file not found: $tmp_file"
        exit 1
    fi

    mkdir -p "$(dirname "$target")"
    mv "$tmp_file" "$target"

    echo "✓ Saved $filename"
}

# --- DMTNet checkpoint ---
download_if_missing \
    "dmtnet.pt" \
    "12oY79kWTLKoIoSkHn8HfM0paw-PClIVV"

# PATNet checkpoint
download_if_missing \
    "patnet.pt" \
    "1jHh7b1gDS-XZIqac-ramN6kcqAnlcHaM"

# --- Label Anything checkpoint ---
download_if_missing \
    "la_5juz3bim.safetensors" \
    "1SARptQ2zLL5pZQILlcOcQj10AK5A23EN"

# HDMNet checkpoints
download_if_missing \
    "hdmnet/coco/split0/resnet50/best_model_5shot.pth" \
    "14TCacpKdHdD4ETPCm_zJNq5C1j0DVR-e"

download_if_missing \
    "hdmnet/coco/split0/resnet50/best_model.pth" \
    "1YU6wnYoQ_YHh5Lh-scKVenvU4-BaCy2J"

# BAM checkpoints
download_if_missing \
    "bam/backbones/resnet50_v2.pth" \
    "1AHcBn0dgcxvlqBwwVupIpjTCzttRM6oV"

download_if_missing \
    "bam/coco/split0/res50/train_epoch_43.5_0.4341.pth" \
    "1NQRwp86Lh9R4BqYuNqQYZg6FCV9HQMtl"

download_if_missing \
    "bam/coco/split0/res50/train5_epoch_47.5_0.4926.pth" \
    "1qyhQuWrqs8-ogHuEK3xh0aOF1I23GWMA"

download_if_missing \
    "bam/PSPNet/coco/split0/resnet50/best.pth" \
    "1Q6bUe8E3ivVxezIb-P-NpErMJMuiktv5"

echo "All checkpoints are ready."