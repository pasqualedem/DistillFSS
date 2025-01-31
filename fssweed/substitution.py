import torch

from einops import rearrange

from fssweed.data.utils import BatchKeys


def cartesian_product(a, b):
    # Create 1D tensors for indices along each dimension
    indices_a = torch.arange(a)
    indices_b = torch.arange(b)

    return torch.cartesian_prod(indices_a, indices_b)


def generate_points_from_errors(
    prediction: torch.tensor,
    ground_truth: torch.tensor,
    num_points: int,
    ignore_index: int = -100,
):
    """
    Generates a point for each class that can be positive or negative depending on the error being false positive or false negative.
    Args:
        prediction (torch.Tensor): The predicted segmentation mask of shape (batch_size, num_classes, height, width)
        ground_truth (torch.Tensor): The ground truth segmentation mask of shape (batch_size, num_classes, height, width)
        num_points (int): The number of points to generate for each class
    """
    B, C = prediction.shape[:2]
    device = prediction.device
    ground_truth = ground_truth.clone()
    ground_truth[ground_truth == ignore_index] = 0
    ground_truth = rearrange(
        torch.nn.functional.one_hot(ground_truth, C),
        "b h w c -> b c h w",
    )
    prediction = prediction.argmax(dim=1)
    prediction = rearrange(
        torch.nn.functional.one_hot(prediction, C),
        "b h w c -> b c h w",
    )
    errors = ground_truth - prediction
    coords = torch.nonzero(errors)
    if coords.shape[0] == 0:
        # No errors
        return (
            torch.zeros(B, C, 1, 2, device=device),
            torch.zeros(B, C, 1, device=device),
        )
    classes, counts = torch.unique(
        coords[:, 0:2], dim=0, return_counts=True, sorted=True
    )
    sampled_idxs = torch.cat(
        [torch.randint(0, x, (num_points,), device=device) for x in counts]
    ) + torch.cat([torch.tensor([0], device=device), counts.cumsum(dim=0)])[
        :-1
    ].repeat_interleave(
        num_points
    )
    sampled_points = coords[sampled_idxs]
    labels = errors[
        sampled_points[:, 0],
        sampled_points[:, 1],
        sampled_points[:, 2],
        sampled_points[:, 3],
    ]
    sampled_points = torch.index_select(
        sampled_points, 1, torch.tensor([0, 1, 3, 2], device=sampled_points.device)
    )  # Swap x and y
    all_classes = cartesian_product(B, C)
    missing = torch.tensor(
        list(
            set(tuple(elem) for elem in all_classes.tolist())
            - set(tuple(elem) for elem in classes.tolist())
        ),
        device=device,
    )
    missing = torch.cat([missing, torch.zeros(missing.shape, device=device)], dim=1)
    sampled_points = torch.cat([sampled_points, missing], dim=0)
    indices = (sampled_points[:, 0] * B + sampled_points[:, 1]).argsort()
    sampled_points = torch.index_select(sampled_points, 0, indices)

    labels = torch.cat([labels, torch.zeros(missing.shape[0], device=device)])
    labels = torch.index_select(labels, 0, indices)

    sampled_points = rearrange(
        sampled_points[:, 2:4],
        "(b c n) xy -> b c n xy",
        n=num_points,
        c=errors.shape[1],
    )
    labels = rearrange(labels, "(b c n) -> b c n", n=num_points, c=errors.shape[1])
    # ignore background
    labels[:, 0] = 0
    return sampled_points, labels


class Substitutor:
    """
    A class that cycle all the images in the examples as a query image.
    """

    torch_keys_to_exchange = [
        BatchKeys.PROMPT_MASKS,
        BatchKeys.FLAG_MASKS,
        BatchKeys.FLAG_EXAMPLES,
        BatchKeys.DIMS,
        BatchKeys.IMAGES,
    ]
    torch_keys_to_separate = [
        BatchKeys.PROMPT_MASKS,
        BatchKeys.FLAG_MASKS,
        BatchKeys.FLAG_EXAMPLES,
    ]
    list_keys_to_exchange = [BatchKeys.CLASSES, BatchKeys.IMAGE_IDS]
    list_keys_to_separate = []

    def __init__(
        self,
        threshold: float = None,
        num_points: int = 1,
        substitute=True,
        subsample=None
    ) -> None:
        self.threshold = threshold
        self.num_points = num_points
        self.substitute = substitute
        self.it = 0
        self.subsample = subsample
        self.num_examples = None

    def reset(self, batch: dict) -> None:
        self.it = 0
        batch, ground_truths = batch
        
        self.ground_truths = ground_truths.clone()
        self.batch = {
            key: value.clone() for key, value in batch.items()
        }
        self.num_examples = self.batch[BatchKeys.IMAGES].shape[1]

    def __iter__(self):
        return self

    def divide_query_examples(self):
        batch_examples = {
            key: self.batch[key][:, 1:] for key in self.torch_keys_to_separate
        }
        batch_examples.update({
            key: [elem[1:] for elem in self.batch[key]] for key in self.list_keys_to_separate
        })
        
        gt = self.ground_truths[:, 0]

        remaining_keys = set(self.batch.keys()) - set(self.torch_keys_to_separate + self.list_keys_to_separate)
        batch_examples.update({key: self.batch[key] for key in remaining_keys})

        support_set_len = self.num_examples - 1

        if self.subsample:
            index_tensor = torch.randperm(support_set_len, device=self.batch["images"].device)[:self.subsample]
            query_index_tensor = torch.cat([torch.tensor([0], device=index_tensor.device), index_tensor + 1])

            for key_set, separate_keys in [(self.torch_keys_to_exchange, self.torch_keys_to_separate),
                                        (self.list_keys_to_exchange, self.list_keys_to_separate)]:
                for key in key_set:
                    if key in batch_examples:
                        indices = index_tensor if key in separate_keys else query_index_tensor
                        if isinstance(batch_examples[key], list):
                            batch_examples[key] = [elem[indices] for elem in batch_examples[key]]
                        else:
                            batch_examples[key] = batch_examples[key][:, indices]

        # Remove classes not present in the support set from GT
        batch_i, class_j = torch.where(batch_examples[BatchKeys.FLAG_EXAMPLES].sum(dim=(1)).logical_not())
        for i in batch_i:
            for j in class_j:
                gt[i][gt[i] == j] = 0
        
        return batch_examples, gt

    def __next__(self):
        device = self.batch["images"].device

        if self.it == 0:
            self.it = 1
            return self.divide_query_examples()
        if not self.substitute:
            raise StopIteration
        if self.it == self.num_examples:
            raise StopIteration
        else:
            query_index = torch.tensor([self.it], device=device)
            remaining_index = torch.cat(
                            [
                                torch.arange(0, self.it, device=device),
                                torch.arange(self.it + 1, self.num_examples, device=device),
                            ]
                        ).long()
        index_tensor = torch.cat([query_index, remaining_index], dim=0)

        for key in self.torch_keys_to_exchange:
            if key in self.batch:
                self.batch[key] = torch.index_select(
                    self.batch[key], dim=1, index=index_tensor
                )

        for key in self.list_keys_to_exchange:
            if key in self.batch:
                self.batch[key] = [
                    [elem[i] for i in index_tensor] for elem in self.batch[key]
                ]
                
        for key in self.batch.keys() - set(
            self.torch_keys_to_exchange + self.list_keys_to_exchange
        ):
            if key in self.batch:
                self.batch[key] = self.batch[key]

        self.ground_truths = torch.index_select(
            self.ground_truths, dim=1, index=index_tensor
        )

        self.it += 1
        return self.divide_query_examples()
