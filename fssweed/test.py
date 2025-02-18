from fssweed.data.utils import BatchKeys
from fssweed.utils.utils import ResultDict, linearize_metrics, to_device
from fssweed.data.utils import merge_dicts


import torch
from tqdm import tqdm


def test(model, dataloader, examples, tracker, logger, dataset_name, image_size, metrics):
    model.eval()
    tracker.log_test_prompts(examples, dataloader.dataset.id2class, dataset_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    id2class = dataloader.dataset.id2class

    bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        postfix={"loss": 0},
        desc="Test: ",
    )
    update_frequency = 10
    tracker.create_image_sequence(dataset_name)
    with torch.no_grad():
        for batch_idx, batch_dict in bar:
            image_dict, gt = batch_dict
            input_dict = merge_dicts(prompts=to_device(examples, device), imgs=to_device(image_dict, device))
            gt = to_device(gt, device)
            outputs = model(input_dict)[ResultDict.LOGITS]
            tracker.log_test_prediction(
                batch_idx=batch_idx,
                input_dict=image_dict,
                gt=gt,
                pred=outputs,
                input_shape=image_size,
                id2classes=dataloader.dataset.id2class,
                dataset_name=dataset_name,
            )
            outputs = torch.argmax(outputs, dim=1)
            dims = image_dict[BatchKeys.DIMS][0].tolist()
            outputs = outputs[:, : dims[0], : dims[1]]
            metrics.update(outputs, gt)
            if batch_idx % update_frequency == 0:
                metrics_values = linearize_metrics(metrics.compute(), id2class=id2class)
                f1_score = metrics_values.get("MulticlassF1Score", 0)
                jaccard = metrics_values.get("MulticlassJaccardIndex", 0)
                bar.set_postfix({"F1": f1_score, "jaccard": jaccard})
        metrics_values = linearize_metrics(metrics.compute(), id2class=id2class)

        tracker.log_metrics(metrics=metrics_values)
        for k, v in metrics_values.items():
            logger.info(f"Test - {k}: {v}")
        tracker.add_image_sequence(dataset_name)