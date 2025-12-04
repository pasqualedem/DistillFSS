import torch

from einops import rearrange

from distillfss.data.utils import BatchKeys


def cartesian_product(a, b):
    # Create 1D tensors for indices along each dimension
    indices_a = torch.arange(a)
    indices_b = torch.arange(b)

    return torch.cartesian_prod(indices_a, indices_b)


def get_substitutor(name, **params):
    if name == "paired":
        return PairedSubstitutor(**params)
    params.pop("iterations_is_num_classes", None)
    return Substitutor(**params)


class Substitutor:
    """
    A class that cycle all the images in the examples as a query image.
    """

    torch_keys_to_exchange = [
        BatchKeys.PROMPT_MASKS,
        BatchKeys.FLAG_EXAMPLES,
        BatchKeys.DIMS,
        BatchKeys.IMAGES,
    ]
    torch_keys_to_separate = [
        BatchKeys.PROMPT_MASKS,
        BatchKeys.FLAG_EXAMPLES,
    ]
    list_keys_to_exchange = [BatchKeys.CLASSES, BatchKeys.IMAGE_IDS]
    list_keys_to_separate = []

    def __init__(
        self,
        threshold: float = None,
        num_points: int = 1,
        substitute=True,
        subsample=None,
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
            device = self.batch["images"].device
            granted_first_sample = torch.tensor([0], device=device)
            index_tensor = torch.randperm(support_set_len-1, device=device)[:self.subsample-1]
            index_tensor = torch.cat([granted_first_sample, index_tensor])
            query_index_tensor = torch.cat([torch.tensor([0], device=device), index_tensor + 1])

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


class PairedSubstitutor(Substitutor):
    def __init__(self, threshold: float = None, num_points: int = 1, substitute=True, subsample=None, 
        iterations_is_num_classes=True,) -> None:
        super().__init__(threshold, num_points, substitute, subsample)
        self.pair_indices = None
        self.num_pairs = None
        self.iterations_is_num_classes = iterations_is_num_classes
    
    def reset(self, batch: dict) -> None:
        super().reset(batch)
        
        # Determine pairs based on FLAG_EXAMPLES
        flag_examples = self.batch[BatchKeys.FLAG_EXAMPLES]  # Shape: [B, M, C]
        B, M, C = flag_examples.shape
        
        if self.iterations_is_num_classes:
            num_iterations = C
        else:
            num_iterations = M
        
        self.pair_indices = []
        for it in range(num_iterations): # Not background
            c = it%(C-1)+1
            example_indices = (flag_examples[:, :, c].sum(dim=0) > 0).nonzero(as_tuple=True)[0]
            if len(example_indices) >= 2:
                example_indices = example_indices[torch.randperm(len(example_indices))]
                self.pair_indices.append(example_indices[:2])
        
        self.num_pairs = len(self.pair_indices)
        self.it = 0
    
    def __next__(self):
        device = self.batch["images"].device
        
        if self.it >= self.num_pairs:
            raise StopIteration
        
        query_idx, second_idx = self.pair_indices[self.it]
        remaining_indices = torch.tensor([
            i for i in range(self.num_examples) if i not in [query_idx, second_idx]
        ], device=device)
        
        index_tensor = torch.cat([
            torch.tensor([query_idx, second_idx], device=device),
            remaining_indices
        ], dim=0)
        
        # Update pair indices
        self.pair_indices = [torch.tensor([torch.where(index_tensor == x)[0].item() for x in paired_idx]) for paired_idx in self.pair_indices]
        
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
        
        self.ground_truths = torch.index_select(
            self.ground_truths, dim=1, index=index_tensor
        )
        
        self.it += 1
        return self.divide_query_examples()
