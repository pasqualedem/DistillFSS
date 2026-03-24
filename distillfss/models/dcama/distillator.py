import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tempfile import TemporaryDirectory as SoftTemporaryDirectory

from distillfss.data.utils import BatchKeys
from distillfss.utils.utils import ResultDict
import yaml

from .dcama import (
    DCAMAMultiClass,
    DCAMA_model,
    boost_coarse_map,
    postprocess_masks,
    refine_coarse_maps,
    reshape_and_prepare_features,
    stack_and_reshape_features,
    build_dcama,
)

from huggingface_hub import (
    EvalResult,
    HfApi,
    ModelCard,
    ModelCardData,
    PyTorchModelHubMixin,
)


class ClassDistiller(nn.Module):
    def __init__(self, dcama: DCAMAMultiClass):
        super().__init__()
        self.coarse_extractor = nn.ModuleList([])
        for inch in dcama.feat_channels[1:]:
            self.coarse_extractor.append(self.build_conv_block(inch))
        self.stack_ids = dcama.stack_ids

    def build_conv_block(self, inch):
        return nn.Sequential(
            nn.Conv2d(inch, inch, kernel_size=3, padding=1),
            # nn.BatchNorm2d(inch),
            nn.ReLU(),
            nn.Conv2d(inch, 1, kernel_size=1),
        )

    def forward(self, query_feats):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            if idx < self.stack_ids[0]:
                continue

            if idx < self.stack_ids[1]:
                coarse = self.coarse_extractor[0](query_feat)
            elif idx < self.stack_ids[2]:
                coarse = self.coarse_extractor[1](query_feat)
            else:
                coarse = self.coarse_extractor[2](query_feat)

            coarse_masks.append(coarse)

        coarse_masks1 = stack_and_reshape_features(
            coarse_masks, self.stack_ids, self.stack_ids[2], self.stack_ids[3]
        )
        coarse_masks2 = stack_and_reshape_features(
            coarse_masks, self.stack_ids, self.stack_ids[1], self.stack_ids[2]
        )
        coarse_masks3 = stack_and_reshape_features(
            coarse_masks, self.stack_ids, self.stack_ids[0], self.stack_ids[1]
        )
        return coarse_masks1, coarse_masks2, coarse_masks3


class DistilledDCAMA(nn.Module, PyTorchModelHubMixin):
    model_card_template = \
        '\n---\n# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1\n# Doc / guide: https://huggingface.co/docs/hub/model-cards\n{{ card_data }}\n---\n\n' \
        'DistillFSS-DCAMA is a distilled version of the DCAMA model for a specific downstream segmentation task. ' \
        'The DistillFSS framework allows to distill large few-shot segmentation models into smaller and more efficient ones, ' \
        'while improving or maintaining their performance on the target task. \n\n' \
        '- Code: {{ repo_url | default("[More Information Needed]", true) }}\n- Paper: {{ paper_url | default("[More Information Needed]", true) }}\n\n' \
        'How to use this model:\n'\
        'Clone the repository: \n```bash\ngit clone https://github.com/pasqualedem/DistillFSS.git\n```\n' \
        'Install the required dependencies as specified in the repository.\n\n' \
        'Load the model using the following code snippet:\n'\
        '```python\nfrom distillfss.models.dcama.distillator import DistilledDCAMA\nmodel = DistilledDCAMA.from_pretrained("{{ repo_id }}")\n```\n\n' \
        'YAML configuration:\n```yaml\n{{ parameters | default("[More Information Needed]", true) }}\n```\n'

    def __init__(self, num_classes, dcama: DCAMAMultiClass = None, dcama_args=None):
        super(DistilledDCAMA, self).__init__()
        self.num_classes = num_classes
        self.teacher = dcama
        self.dcama_args = dcama_args
        if dcama_args is not None and dcama is not None:
            raise ValueError("You should provide either dcama or dcama_args, not both.")
        if dcama is None and dcama_args is not None:
            self.teacher = build_dcama(**dcama_args)
        if dcama is None and dcama_args is None:
            raise ValueError("You should provide either dcama or dcama_args.")

        self.student = nn.ModuleList()

        for i in range(num_classes):
            self.student.append(ClassDistiller(self.teacher))

    def extract_features(self, x):
        query = x[BatchKeys.IMAGES][:, 0]
        query_features = self.teacher.extract_feats(query)
        return {ResultDict.QUERY_FEATS: query_features, ResultDict.SUPPORT_FEATS: None}

    def forward(self, x):
        if self.training:
            dcama_result = self.teacher(x)
        else:
            dcama_result = self.extract_features(x)

        dcama_result[ResultDict.DISTILLED_COARSE] = []
        dcama_result[ResultDict.DISTILLED_LOGITS] = []
        for i, distiller in enumerate(self.student):
            query_feats = dcama_result[ResultDict.QUERY_FEATS]
            support_feats = dcama_result[ResultDict.SUPPORT_FEATS]
            if query_feats[i] is None:
                query_img = x[BatchKeys.IMAGES][:, 0]
                dcama_result[ResultDict.DISTILLED_COARSE].append(None)
                dcama_result[ResultDict.DISTILLED_LOGITS].append(
                    torch.full(
                        (1, 2, *query_img.shape[-2:]),
                        -torch.inf,
                        device=query_img.device,
                    )
                )
                continue
            if isinstance(
                query_feats[i], list
            ):  # If the element is still a list and not a tensor
                query_feats = query_feats[i]
                support_feats = support_feats[i]

            coarse_masks1, coarse_masks2, coarse_masks3 = distiller(query_feats)
            mix = self.teacher.model.mix_maps(
                coarse_masks1, coarse_masks2, coarse_masks3
            )
            mix, _, _ = self.teacher.model.skip_concat_features(
                mix, query_feats, support_feats, None
            )
            logit_mask, _, _ = self.teacher.model.upsample_and_classify(mix)
            dcama_result[ResultDict.DISTILLED_COARSE].append(
                [coarse_masks1, coarse_masks2, coarse_masks3]
            )
            dcama_result[ResultDict.DISTILLED_LOGITS].append(logit_mask)

        logits = dcama_result[ResultDict.DISTILLED_LOGITS]
        logits = torch.stack(logits, dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = postprocess_masks(logits, x["dims"])

        key = ResultDict.DISTILLED_LOGITS if self.training else ResultDict.LOGITS

        return {
            **dcama_result,
            key: logits,
        }

    def push_to_hub(
        self,
        repo_id,
        *,
        commit_message="Push model using huggingface_hub.",
        private=None,
        token=None,
        branch=None,
        create_pr=None,
        allow_patterns=None,
        ignore_patterns=None,
        delete_patterns=None,
        model_card_kwargs=None,
        parameters=None,
    ):

        model_card_kwargs = {
            "name": "DistillFSS-DCAMA",
            "tags": ["few-shot segmentation", "distillation", "image-segmentation"],
            "license": "mit",
            "language": "en",
            "library": "pytorch",
            "ArXiv": "2512.05613",
            "base_model": "DCAMA",
            "repo_url": "https://github.com/pasqualedem/DistillFSS",
            "paper_url": "https://arxiv.org/abs/2512.05613",
            "parameters": yaml.dump(parameters) if parameters else None,
            "repo_id": repo_id,
            
            **(model_card_kwargs or {}),
        }
        api = HfApi(token=token)
        repo_id = api.create_repo(
            repo_id=repo_id, private=private, exist_ok=True
        ).repo_id

        # Push the files to the repo in a single commit
        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id

            # save model card
            model_card_path = saved_path / "README.md"
            model_card_kwargs = (
                model_card_kwargs if model_card_kwargs is not None else {}
            )

            card_data = ModelCardData(
                # eval_results=[
                #     EvalResult(
                #         task_type="image-classification",
                #         dataset_type="beans",
                #         dataset_name="Beans",
                #         metric_type="accuracy",
                #         metric_value=0.9,
                #     ),
                # ],
                **model_card_kwargs,
            )
            card = ModelCard.from_template(
                card_data,
                template_str=self.model_card_template,
            )
            card.save(model_card_path)

            dcama_args = {k: v for k, v in parameters["model"]["params"]["teacher"].items() if k != "name"}
            self.save_pretrained(
                saved_path, config={"num_classes": self.num_classes, "dcama_args": dcama_args}
            )
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )


class SupportDistillerBlock(nn.Module):
    def __init__(self, inch, heads=32, use_support=False):
        super().__init__()

        self.keys = nn.Embedding(heads, inch)
        self.heads = heads
        self.use_support = use_support
        if use_support:
            self.conv_mappings = nn.ModuleList(
                [self.build_conv_block(inch) for _ in range(heads)]
            )
        self.attn = nn.MultiheadAttention(inch, num_heads=8)
        self.out = nn.Conv2d(inch, 1, kernel_size=1)

    def build_conv_block(self, inch):
        return nn.Sequential(nn.Conv2d(inch, inch, kernel_size=1, padding=1))

    def forward(self, query_feats, support_feats):
        if self.use_support:
            mapped_supports = [
                self.conv_mappings[i](support_feats) for i in range(self.heads)
            ]
            mapped_support = torch.stack(mapped_supports, dim=1)

            # Average pooling to remove H and W dimension
            mapped_support = einops.reduce(
                mapped_support, "b heads c h w -> heads b c", "mean"
            )
            keys = einops.repeat(
                self.keys.weight, "heads c -> heads b c", b=query_feats.shape[0]
            )
            keys = keys + mapped_support
        else:
            keys = einops.repeat(
                self.keys.weight, "heads c -> heads b c", b=query_feats.shape[0]
            )

        h, w = query_feats.shape[-2:]
        query_feats = einops.rearrange(query_feats, "b c h w -> (h w) b c")

        coarse_map = self.attn(query_feats, keys, keys)[0]
        coarse_map = einops.rearrange(coarse_map, "(h w) b c -> b c h w", h=h, w=w)
        coarse_map = self.out(coarse_map)
        return coarse_map


class SupportDistiller(nn.Module):
    def __init__(self, dcama: DCAMAMultiClass, use_support=False):
        super().__init__()
        self.coarse_extractor = nn.ModuleList([])
        for inch in dcama.feat_channels[1:]:
            self.coarse_extractor.append(
                SupportDistillerBlock(inch, use_support=use_support)
            )
        self.stack_ids = dcama.stack_ids

    def forward(self, query_feats, support_feats):
        coarse_masks = []
        for idx, query_feat in enumerate(query_feats):
            if support_feats is None:
                support_feat = None
            elif isinstance(support_feats[0], list):
                support_feat = torch.cat(
                    [support_example[idx] for support_example in support_feats], dim=2
                )
            else:
                support_feat = support_feats[idx]

            if idx < self.stack_ids[0]:
                continue
            if idx < self.stack_ids[1]:
                coarse = self.coarse_extractor[0](query_feat, support_feat)
            elif idx < self.stack_ids[2]:
                coarse = self.coarse_extractor[1](query_feat, support_feat)
            else:
                coarse = self.coarse_extractor[2](query_feat, support_feat)

            coarse_masks.append(coarse)

        coarse_masks1 = stack_and_reshape_features(
            coarse_masks, self.stack_ids, self.stack_ids[2], self.stack_ids[3]
        )
        coarse_masks2 = stack_and_reshape_features(
            coarse_masks, self.stack_ids, self.stack_ids[1], self.stack_ids[2]
        )
        coarse_masks3 = stack_and_reshape_features(
            coarse_masks, self.stack_ids, self.stack_ids[0], self.stack_ids[1]
        )
        return coarse_masks1, coarse_masks2, coarse_masks3


class AttentionDistilledDCAMA(nn.Module):
    def __init__(self, num_classes, dcama: DCAMAMultiClass, use_support=False):
        super().__init__()
        self.num_classes = num_classes
        self.teacher = dcama
        self.use_support = use_support
        self.support_feats = None

        self.student = nn.ModuleList()

        for i in range(num_classes):
            self.student.append(SupportDistiller(dcama, use_support=use_support))

    def extract_features(self, x):
        query = x[BatchKeys.IMAGES][:, 0]
        query_features = self.teacher.extract_feats(query)
        if not self.training and self.support_feats is None:
            support = x[BatchKeys.IMAGES][:, 1:]
            support_features = [
                self.teacher.extract_feats(support[:, i])
                for i in range(support.shape[1])
            ]
            self.support_feats = support_features
        return {
            ResultDict.QUERY_FEATS: query_features,
            ResultDict.SUPPORT_FEATS: self.support_feats,
        }

    def forward(self, x):
        if self.training:
            dcama_result = self.teacher(x)
        else:
            dcama_result = self.extract_features(x)

        dcama_result[ResultDict.DISTILLED_COARSE] = []
        dcama_result[ResultDict.DISTILLED_LOGITS] = []
        for i, distiller in enumerate(self.student):
            query_feats = dcama_result[ResultDict.QUERY_FEATS]
            support_feats = dcama_result[ResultDict.SUPPORT_FEATS]
            if query_feats[i] is None:
                query_img = x[BatchKeys.IMAGES][:, 0]
                dcama_result[ResultDict.DISTILLED_COARSE].append(None)
                dcama_result[ResultDict.DISTILLED_LOGITS].append(
                    torch.full(
                        (1, 2, *query_img.shape[-2:]),
                        -torch.inf,
                        device=query_img.device,
                    )
                )
                continue
            if isinstance(
                query_feats[i], list
            ):  # If the element is still a list and not a tensor
                query_feats = query_feats[i]
                support_feats = support_feats[i]

            coarse_masks1, coarse_masks2, coarse_masks3 = distiller(
                query_feats, support_feats
            )
            mix = self.teacher.model.mix_maps(
                coarse_masks1, coarse_masks2, coarse_masks3
            )
            mix, _, _ = self.teacher.model.skip_concat_features(
                mix, query_feats, support_feats, None
            )
            logit_mask, _, _ = self.teacher.model.upsample_and_classify(mix)
            dcama_result[ResultDict.DISTILLED_COARSE].append(
                [coarse_masks1, coarse_masks2, coarse_masks3]
            )
            dcama_result[ResultDict.DISTILLED_LOGITS].append(logit_mask)

        logits = dcama_result[ResultDict.DISTILLED_LOGITS]
        logits = torch.stack(logits, dim=1)
        fg_logits = logits[:, :, 1, ::]
        bg_logits = logits[:, :, 0, ::]
        bg_positions = fg_logits.argmax(dim=1)
        bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
        logits = torch.cat([bg_logits, fg_logits], dim=1)

        logits = postprocess_masks(logits, x["dims"])

        key = ResultDict.DISTILLED_LOGITS if self.training else ResultDict.LOGITS

        return {
            **dcama_result,
            key: logits,
        }
