import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
sys.path.extend(["..", "."])
import numpy as np
from numpy.testing import assert_allclose
import torch

dim_feature = 64
initial_alpha = np.random.randn()
initial_beta = np.random.randn()
x = torch.randn(dim_feature)

def test_shifted_softplus():
    from enerzyme.models.activation import ShiftedSoftplus as F1
    from spookynet.modules.shifted_softplus import ShiftedSoftplus as F2
    f1 = F1(dim_feature, initial_alpha, initial_beta)
    f2 = F2(dim_feature, initial_alpha, initial_beta)
    assert_allclose(f1(x).detach().numpy(), f2(x).detach().numpy())