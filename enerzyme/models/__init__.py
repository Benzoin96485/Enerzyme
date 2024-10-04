from torch import Tensor
from .modelhub import ModelHub, get_model_str, get_pretrain_path
from .ff import FF_single, SEP, FF_REGISTER, build_model, FF_committee


USE_SEGMENT_COO = False


if USE_SEGMENT_COO:
    from torch_scatter import segment_sum_coo
else:
    from torch_scatter import scatter_sum
    def segment_sum_coo(src: Tensor, idx: Tensor, dim_size: int) -> Tensor:
        return scatter_sum(src, idx, dim=idx.dim() - 1, dim_size=dim_size)
