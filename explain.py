from ffss.data.coco import CocoLVISCrop
from ffss.data.utils import imagenet_id2label
from ffss.models import build_model


def explain_coco(
    instances_path="../datasets/coco/annotations/instances_train2017.json",
    img_dir="../datasets/coco/train2017",
    model="resnet50",
):
    model, processor = build_model(model)
    id2label = imagenet_id2label()

    coco_crop = CocoLVISCrop(
        instances_path=instances_path, img_dir=img_dir, preprocess=processor
    )
    
    
