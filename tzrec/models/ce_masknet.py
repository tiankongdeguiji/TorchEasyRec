# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

import torch
import torchmetrics
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.metrics import recall_at_k
from tzrec.models.rank_model import RankModel
from tzrec.modules.masknet import MaskNetModule
from tzrec.protos.loss_pb2 import LossConfig
from tzrec.protos.metric_pb2 import MetricConfig
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class MixCEMaskNet(RankModel):
    """Masknet with MixCrossEntropy Loss for negative sample.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_dim = self.embedding_group.group_total_dim(self.group_name)

        masknet_config = self._model_config.mask_net
        self._sample_num = self._model_config.sample_num

        self.mask_net_layer = MaskNetModule(
            feature_dim, **config_to_kwargs(masknet_config)
        )
        self.output_linear = nn.Linear(
            masknet_config.top_mlp.hidden_units[-1], self._num_class, bias=False
        )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward method."""
        feature_dict = self.build_input(batch)
        features = feature_dict[self.group_name]

        hidden = self.mask_net_layer(features)

        output = self.output_linear(hidden)
        return self._output_to_prediction(output)

    def _output_to_prediction_impl(
        self,
        output: torch.Tensor,
        loss_cfg: LossConfig,
        num_class: int = 1,
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        predictions = {}
        loss_type = loss_cfg.WhichOneof("loss")
        if loss_type == "mix_cross_entropy":
            assert num_class == 1, f"num_class must be 1 when loss type is {loss_type}"
            output = torch.squeeze(output, dim=1)
            predictions["logits" + suffix] = output
            predictions["probs" + suffix] = torch.sigmoid(output)
        else:
            raise NotImplementedError
        return predictions

    def _init_loss_impl(
        self,
        loss_cfg: LossConfig,
        num_class: int = 1,
        reduction: str = "none",
        suffix: str = "",
    ) -> None:
        loss_type = loss_cfg.WhichOneof("loss")

        if loss_type == "mix_cross_entropy":
            loss_name = "binary_cross_entropy" + suffix
            self._loss_modules[loss_name] = nn.BCEWithLogitsLoss(reduction=reduction)

            loss_name = "softmax_cross_entropy" + suffix
            self._loss_modules[loss_name] = nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise ValueError(f"loss[{loss_type}] is not supported yet.")

    def _loss_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label: torch.Tensor,
        loss_weight: Optional[torch.Tensor],
        loss_cfg: LossConfig,
        num_class: int = 1,
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        loss_type = loss_cfg.WhichOneof("loss")

        if loss_type == "mix_cross_entropy":
            loss_name = "binary_cross_entropy" + suffix
            batch_size = label.size(0)
            pred = predictions["logits" + suffix][:batch_size]
            losses[loss_name] = self._loss_modules[loss_name](
                pred, label.to(torch.float32)
            )

            loss_name = "softmax_cross_entropy" + suffix
            pred = predictions["logits" + suffix]
            pred = torch.cat(
                [
                    pred[:batch_size].unsqueeze(1),
                    pred[batch_size:].reshape(-1, self._sample_num),
                ],
                dim=1,
            )
            label = torch.zeros_like(label)
            losses[loss_name] = self._loss_modules[loss_name](pred, label)
        else:
            raise ValueError(f"loss[{loss_type}] is not supported yet.")
        if loss_weight is not None:
            losses[loss_name] = torch.mean(losses[loss_name] * loss_weight)
        return losses

    def _init_metric_impl(
        self, metric_cfg: MetricConfig, num_class: int = 1, suffix: str = ""
    ) -> None:
        metric_type = metric_cfg.WhichOneof("metric")
        metric_name = metric_type + suffix
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        metric_kwargs = config_to_kwargs(oneof_metric_cfg)
        assert num_class == 1
        if metric_type == "auc":
            metric_name = metric_type + suffix
            self._metric_modules[metric_name] = torchmetrics.AUROC(
                task="binary", **metric_kwargs
            )
        elif metric_type == "recall_at_k":
            metric_name = f"recall@{oneof_metric_cfg.top_k}" + suffix
            self._metric_modules[metric_name] = recall_at_k.RecallAtK(**metric_kwargs)
        else:
            raise ValueError(f"{metric_type} is not supported for this model")

    def _update_metric_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label: torch.Tensor,
        metric_cfg: MetricConfig,
        num_class: int = 1,
        suffix: str = "",
    ) -> None:
        metric_type = metric_cfg.WhichOneof("metric")
        metric_name = metric_type + suffix
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        assert num_class == 1
        if metric_type == "auc":
            batch_size = label.size(0)
            pred = predictions["probs" + suffix][:batch_size]
            self._metric_modules[metric_name].update(pred, label)
        elif metric_type == "recall_at_k":
            metric_name = f"recall@{oneof_metric_cfg.top_k}" + suffix
            pred = predictions["probs" + suffix]
            pred = torch.cat(
                [
                    pred[:batch_size].unsqueeze(1),
                    pred[batch_size:].reshape(-1, self._sample_num),
                ],
                dim=1,
            )
            label = torch.zeros_like(pred, dtype=torch.bool)
            label[:, 0] = True
            self._metric_modules[metric_name].update(pred, label)
        else:
            raise ValueError(f"{metric_type} is not supported for this model")
