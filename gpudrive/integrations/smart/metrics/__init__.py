# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from gpudrive.integrations.smart.metrics.cross_entropy import CrossEntropy
from gpudrive.integrations.smart.metrics.ego_nll import EgoNLL
from gpudrive.integrations.smart.metrics.gmm_ade import GMMADE
from gpudrive.integrations.smart.metrics.min_ade import minADE
from gpudrive.integrations.smart.metrics.next_token_cls import TokenCls
# from gpudrive.integrations.smart.metrics.wosac_metrics import WOSACMetrics
# from gpudrive.integrations.smart.metrics.wosac_submission import WOSACSubmission
from gpudrive.integrations.smart.modules.smart_decoder import SMARTDecoder
from gpudrive.integrations.smart.tokens.token_processor import TokenProcessor
from gpudrive.integrations.smart.utils.finetune import set_model_for_finetuning
from gpudrive.integrations.smart.layers.attention_layer import AttentionLayer
from gpudrive.integrations.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from gpudrive.integrations.smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
from gpudrive.integrations.smart.metrics.utils import get_euclidean_targets, get_prob_targets
import torch
import torch.nn as nn

__all__ = [
    'CrossEntropy', 'EgoNLL', 'GMMADE', 'minADE', 'TokenCls',
    # 'WOSACMetrics', 'WOSACSubmission'
]
