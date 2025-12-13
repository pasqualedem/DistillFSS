CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

download_if_missing () {
    local filename="$1"
    local file_id="$2"
    local target="$CHECKPOINT_DIR/$filename"

    if [ -f "$target" ]; then
        echo "✓ $filename already exists"
    else
        echo "↓ Downloading $filename"
        gdown "$file_id" || { echo "Download failed for $filename"; exit 1; }

        if [ ! -f "$filename" ]; then
            echo "Downloaded file not found: $filename"
            exit 1
        fi

        mv "$filename" "$target"
        echo "✓ Saved $filename"
    fi
}

# --- DCAMA SWIN checkpoint ---
download_if_missing \
    "swin_fold0_pascal_modcross_soft.safetensors" \
    "1J45U_rh-QnrFsS_YkBs9zR0GqGsQP2en"

# --- Backbone checkpoints ---
download_if_missing \
    "swin_base_patch4_window12_384.pth" \
    "1NlX0IFgcrjdHmbTBrXrqAdzYJ7Ul9ux6"

download_if_missing \
    "resnet50_a1h-35c100f8.pth" \
    "1eLzflyB1gfhYHw6dYKnSQfZP3OIiM_Uj"

download_if_missing \
    "resnet50_fold0_pascal_modcross_soft.pt" \
    "1Z6RKkOmqeEKmpEhCY9CVjJS8ykUEaC7D"
