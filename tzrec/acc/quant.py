# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Type

import torch
import torch.nn as nn
import torch.quantization as quant
import torchrec as trec
import torchrec.quant.embedding_modules as trec_quant
from torchrec.modules.embedding_configs import QuantConfig


def quantize_embeddings(
    module: nn.Module,
    dtype: torch.dtype,
    inplace: bool,
    additional_qconfig_spec_keys: Optional[List[Type[nn.Module]]] = None,
    additional_mapping: Optional[Dict[Type[nn.Module], Type[nn.Module]]] = None,
    output_dtype: torch.dtype = torch.float,
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None,
) -> nn.Module:
    """Quant Embedding."""
    qconfig = QuantConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_dtype),
        weight=quant.PlaceholderObserver.with_args(dtype=dtype),
        per_table_weight_dtype=per_table_weight_dtype,
    )
    qconfig_spec: Dict[Type[nn.Module], QuantConfig] = {
        trec.EmbeddingBagCollection: qconfig,
        trec.EmbeddingCollection: qconfig,
    }
    mapping: Dict[Type[nn.Module], Type[nn.Module]] = {
        trec.EmbeddingBagCollection: trec_quant.EmbeddingBagCollection,
        trec.EmbeddingCollection: trec_quant.EmbeddingCollection,
    }
    if additional_qconfig_spec_keys is not None:
        for t in additional_qconfig_spec_keys:
            qconfig_spec[t] = qconfig
    if additional_mapping is not None:
        mapping.update(additional_mapping)
    return quant.quantize_dynamic(
        module,
        qconfig_spec=qconfig_spec,
        mapping=mapping,
        inplace=inplace,
    )
