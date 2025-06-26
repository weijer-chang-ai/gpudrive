# SMART Integration for GPUDrive
# This module provides SMART (Scalable Multiagent Autonomous Reasoning with Transformers) integration

from gpudrive.integrations.smart.model import SMART, EgoGMMSMART
from gpudrive.integrations.smart.modules import (
    SMARTDecoder, EgoGMMSMARTDecoder, SMARTMapDecoder, 
    SMARTAgentDecoder, EgoGMMAgentDecoder
)
from gpudrive.integrations.smart.tokens import TokenProcessor
from gpudrive.integrations.smart.layers import (
    AttentionLayer, FourierEmbedding, MLPEmbedding, MLPLayer
)
from gpudrive.integrations.smart.metrics import (
    CrossEntropy, EgoNLL, GMMADE, minADE, TokenCls,
    # WOSACMetrics, WOSACSubmission
)
from gpudrive.integrations.smart.utils import (
    angle_between_2d_vectors, wrap_angle,
    cal_polygon_contour, sample_next_gmm_traj, sample_next_token_traj,
    transform_to_global, transform_to_local, weight_init
)

__all__ = [
    # Models
    'SMART', 'EgoGMMSMART',
    # Modules
    'SMARTDecoder', 'EgoGMMSMARTDecoder', 'SMARTMapDecoder', 
    'SMARTAgentDecoder', 'EgoGMMAgentDecoder',
    # Tokens
    'TokenProcessor',
    # Layers
    'AttentionLayer', 'FourierEmbedding', 'MLPEmbedding', 'MLPLayer',
    # Metrics
    'CrossEntropy', 'EgoNLL', 'GMMADE', 'minADE', 'TokenCls',
    # 'WOSACMetrics', 'WOSACSubmission',
    # Utils
    'angle_between_2d_vectors', 'wrap_angle',
    'cal_polygon_contour', 'sample_next_gmm_traj', 'sample_next_token_traj',
    'transform_to_global', 'transform_to_local', 'weight_init'
]
