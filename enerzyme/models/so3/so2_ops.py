import torch
import torch.nn as nn
import math
import copy
from typing import List, Optional

from torch.nn import Linear
from .embedding import SO3_Embedding
from ..blocks.radial_mlp import RadialFunction


class SO2_m_Convolution(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
    """
    def __init__(
        self,
        m, 
        sphere_channels,
        m_output_channels,
        lmax_list, 
        mmax_list
    ):
        super(SO2_m_Convolution, self).__init__()
        
        self.m = m
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        num_channels = 0
        for i in range(self.num_resolutions):
            num_coefficents = 0
            if self.mmax_list[i] >= self.m:
                num_coefficents = self.lmax_list[i] - self.m + 1
            num_channels = num_channels + num_coefficents * self.sphere_channels
        assert num_channels > 0

        self.fc = Linear(num_channels, 
            2 * self.m_output_channels * (num_channels // self.sphere_channels), 
            bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))


    def forward(self, x_m):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.fc.out_features // 2)
        x_i = x_m.narrow(2, self.fc.out_features // 2, self.fc.out_features // 2)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1) #x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1) #x_r[:, 1] + x_i[:, 0]
        x_out = torch.cat((x_m_r, x_m_i), dim=1)
        
        return x_out


class SO2_Convolution(torch.nn.Module):
    """
    SO(2) Block: Perform SO(2) convolutions for all m (orders)

    Args:
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
        mappingReduced (CoefficientMappingModule): Used to extract a subset of m components
        internal_weights (bool):    If True, not using radial function to multiply inputs features
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
        extra_m0_output_channels (int): If not None, return `out_embedding` (SO3_Embedding) and `extra_m0_features` (Tensor).
    """
    def __init__(
        self,
        sphere_channels,
        m_output_channels,
        lmax_list,
        mmax_list,
        mappingReduced,
        internal_weights=True,
        edge_channels_list=None,
        extra_m0_output_channels=None
    ):
        super(SO2_Convolution, self).__init__()
        self.sphere_channels = sphere_channels
        self.m_output_channels = m_output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.mappingReduced = mappingReduced
        self.num_resolutions = len(lmax_list)
        self.internal_weights = internal_weights
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.extra_m0_output_channels = extra_m0_output_channels

        num_channels_rad = 0    # for radial function

        num_channels_m0 = 0
        for i in range(self.num_resolutions):
            num_coefficients = self.lmax_list[i] + 1
            num_channels_m0 = num_channels_m0 + num_coefficients * self.sphere_channels

        # SO(2) convolution for m = 0
        m0_output_channels = self.m_output_channels * (num_channels_m0 // self.sphere_channels)
        if self.extra_m0_output_channels is not None:
            m0_output_channels = m0_output_channels + self.extra_m0_output_channels
        self.fc_m0 = Linear(num_channels_m0, m0_output_channels)
        num_channels_rad = num_channels_rad + self.fc_m0.in_features
        
        # SO(2) convolution for non-zero m
        self.so2_m_conv = nn.ModuleList()
        for m in range(1, max(self.mmax_list) + 1):
            self.so2_m_conv.append(
                SO2_m_Convolution(
                    m, 
                    self.sphere_channels,
                    self.m_output_channels,
                    self.lmax_list, 
                    self.mmax_list,
                )
            )
            num_channels_rad = num_channels_rad + self.so2_m_conv[-1].fc.in_features

        # Embedding function of distance
        self.rad_func = None
        if not self.internal_weights:
            assert self.edge_channels_list is not None
            self.edge_channels_list.append(int(num_channels_rad))
            self.rad_func = RadialFunction(self.edge_channels_list)


    def forward(self, x, x_edge):

        num_edges = len(x_edge)
        out = []

        # Reshape the spherical harmonics based on m (order)
        x._m_primary(self.mappingReduced)

        # radial function
        if self.rad_func is not None:
            x_edge = self.rad_func(x_edge)
        offset_rad = 0

        # Compute m=0 coefficients separately since they only have real values (no imaginary)
        x_0 = x.embedding.narrow(1, 0, self.mappingReduced.m_size[0])
        x_0 = x_0.reshape(num_edges, -1)
        if self.rad_func is not None:
            x_edge_0 = x_edge.narrow(1, 0, self.fc_m0.in_features)
            x_0 = x_0 * x_edge_0
        x_0 = self.fc_m0(x_0)

        x_0_extra = None
        # extract extra m0 features 
        if self.extra_m0_output_channels is not None:
            x_0_extra = x_0.narrow(-1, 0, self.extra_m0_output_channels)
            x_0 = x_0.narrow(-1, self.extra_m0_output_channels, (self.fc_m0.out_features - self.extra_m0_output_channels))
        
        x_0 = x_0.view(num_edges, -1, self.m_output_channels)
        #x.embedding[:, 0 : self.mappingReduced.m_size[0]] = x_0
        out.append(x_0)
        offset_rad = offset_rad + self.fc_m0.in_features

        # Compute the values for the m > 0 coefficients
        offset = self.mappingReduced.m_size[0]
        for m in range(1, max(self.mmax_list) + 1):
            # Get the m order coefficients
            x_m = x.embedding.narrow(1, offset, 2 * self.mappingReduced.m_size[m])
            x_m = x_m.reshape(num_edges, 2, -1)

            # Perform SO(2) convolution
            if self.rad_func is not None:
                x_edge_m = x_edge.narrow(1, offset_rad, self.so2_m_conv[m - 1].fc.in_features)
                x_edge_m = x_edge_m.reshape(num_edges, 1, self.so2_m_conv[m - 1].fc.in_features)
                x_m = x_m * x_edge_m
            x_m = self.so2_m_conv[m - 1](x_m)
            x_m = x_m.view(num_edges, -1, self.m_output_channels)
            #x.embedding[:, offset : offset + 2 * self.mappingReduced.m_size[m]] = x_m
            out.append(x_m)
            offset = offset + 2 * self.mappingReduced.m_size[m]
            offset_rad = offset_rad + self.so2_m_conv[m - 1].fc.in_features

        out = torch.cat(out, dim=1)
        out_embedding = SO3_Embedding(
            0, 
            x.lmax_list.copy(), 
            self.m_output_channels, 
            device=x.device, 
            dtype=x.dtype
        )
        out_embedding.set_embedding(out)
        out_embedding.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # Reshape the spherical harmonics based on l (degree)
        out_embedding._l_primary(self.mappingReduced)

        if self.extra_m0_output_channels is not None:
            return out_embedding, x_0_extra
        else:
            return out_embedding



class SO2MLinear(torch.nn.Module):
    """SO(2) linear on +-m features (EquiformerV3 fused path).

    Expects m-primary layout from ``SO3RotationFused``. Distinct from
    ``SO2_m_Convolution`` used by EquiformerV2.
    """

    def __init__(self, m, num_in_channels, num_out_channels, lmax, mmax):
        super().__init__()
        self.m = m
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        num_m_components = self.lmax - self.m + 1
        assert num_m_components > 0
        self.in_features = num_m_components * self.num_in_channels
        self.out_features = num_m_components * self.num_out_channels
        self.fc = Linear(self.in_features, (2 * self.out_features), bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m, concat_outputs=True):
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.out_features)
        x_i = x_m.narrow(2, self.out_features, self.out_features)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)
        x_out = (x_m_r, x_m_i)
        if concat_outputs:
            x_out = torch.cat(x_out, dim=1)
        return x_out


class SO2Linear(torch.nn.Module):
    """SO(2) linear over all m (EquiformerV3).

    Input layout is m-primary: (0,...), (1,...), ... as produced by
    ``SO3RotationFused.rotate``.
    """

    def __init__(
        self,
        num_in_channels,
        num_out_channels,
        lmax,
        mmax,
        extra_m0_out_channels=None,
    ):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.lmax = lmax
        self.mmax = mmax
        self.extra_m0_out_channels = extra_m0_out_channels

        num_in_channels_m0 = (self.lmax + 1) * self.num_in_channels
        num_out_channels_m0 = (self.lmax + 1) * self.num_out_channels
        if self.extra_m0_out_channels is not None:
            self.num_channels_m0_list = [self.extra_m0_out_channels, num_out_channels_m0]
            num_out_channels_m0 = num_out_channels_m0 + self.extra_m0_out_channels
        self.fc_m0 = Linear(num_in_channels_m0, num_out_channels_m0)

        self.so2_m_linear = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.so2_m_linear.append(
                SO2MLinear(
                    m,
                    self.num_in_channels,
                    self.num_out_channels,
                    self.lmax,
                    self.mmax,
                )
            )

    def forward(self, x):
        num_edges = x.shape[0]
        outputs = []
        x_m0 = x.narrow(1, 0, (self.lmax + 1))
        x_m0 = x_m0.reshape(num_edges, -1)
        x_m0 = self.fc_m0(x_m0)
        x_m0_extra = None
        if self.extra_m0_out_channels is not None:
            x_m0_extra, x_m0 = torch.split(x_m0, self.num_channels_m0_list, dim=1)
        x_m0 = x_m0.view(num_edges, -1, self.num_out_channels)
        outputs.append(x_m0)
        offset = self.lmax + 1
        for m in range(1, self.mmax + 1):
            x_m = x.narrow(1, offset, 2 * (self.lmax + 1 - m))
            offset = offset + 2 * (self.lmax + 1 - m)
            x_m = x_m.reshape(num_edges, 2, -1)
            x_m = self.so2_m_linear[m - 1](x_m, concat_outputs=False)
            x_m_pos, x_m_neg = x_m[0], x_m[1]
            outputs.append(x_m_pos.view(num_edges, -1, self.num_out_channels))
            outputs.append(x_m_neg.view(num_edges, -1, self.num_out_channels))
        outputs = torch.cat(outputs, dim=1)
        if self.extra_m0_out_channels is not None:
            return outputs, x_m0_extra
        return outputs


class uvSO2MLinear(torch.nn.Module):
    """uv SO(2) linear on +-m (TECE / TACE). Supports w1_w2 / w1_w1 / w1."""

    def __init__(
        self,
        m: int,
        num_channel_in: int,
        num_channel_out: int,
        num_components_in: int,
        num_components_out: int,
        weight_type: str = "w1_w2",
    ):
        super().__init__()
        self.m = m
        self.num_channel_in = num_channel_in
        self.num_channel_out = num_channel_out
        self.num_components_in = num_components_in
        self.num_components_out = num_components_out
        self.weight_type = weight_type
        assert self.num_components_in > 0
        assert self.num_components_out > 0

        in_f = self.num_components_in * self.num_channel_in
        out_f = self.num_components_out * self.num_channel_out
        if weight_type == "w1_w2":
            self.fc = Linear(in_f, out_f * 2, bias=False)
            self.fc.weight.data.mul_(1 / math.sqrt(2))
        else:
            self.fc = Linear(in_f, out_f, bias=False)
            if weight_type == "w1_w1":
                self.fc.weight.data.mul_(1 / math.sqrt(2))
        self._Cout = out_f

    def forward(self, x, concat_outputs=True):
        if self.weight_type == "w1_w2":
            return self._w1_w2_forward(x, concat_outputs)
        if self.weight_type == "w1_w1":
            return self._w1_w1_forward(x, concat_outputs)
        return self._w1_forward(x, concat_outputs)

    def _w1_w2_forward(self, x, concat_outputs=True):
        x = self.fc(x)
        w1_x = x.narrow(2, 0, self._Cout)
        w2_x = x.narrow(2, self._Cout, self._Cout)
        xr = w1_x.narrow(1, 0, 1) - w2_x.narrow(1, 1, 1)
        xi = w1_x.narrow(1, 1, 1) + w2_x.narrow(1, 0, 1)
        x_out = (xr, xi)
        if concat_outputs:
            return torch.cat(x_out, dim=1)
        return x_out

    def _w1_w1_forward(self, x, concat_outputs=True):
        xr = x.narrow(1, 0, 1)
        xi = x.narrow(1, 1, 1)
        yr = self.fc(xr - xi)
        yi = self.fc(xi + xr)
        x_out = (yr, yi)
        if concat_outputs:
            return torch.cat(x_out, dim=1)
        return x_out

    def _w1_forward(self, x, concat_outputs=True):
        x = self.fc(x)
        if concat_outputs:
            return x
        return (x.narrow(1, 0, 1), x.narrow(1, 1, 1))


class uvSO2Linear(torch.nn.Module):
    """uv SO(2) linear over all m (TECE / TACE).

    Distinct from EquiformerV3 :class:`SO2Linear` (extra_m0 API). Prefer this for
    TECE ECE / RRA paths that need ``weight_type`` and custom component counts.
    """

    def __init__(
        self,
        mmax: int,
        lmax: int,
        num_channel_in: int,
        num_channel_out: int,
        num_components_in: Optional[List[int]] = None,
        num_components_out: Optional[List[int]] = None,
        weight_type: str = "w1_w2",
    ):
        super().__init__()
        self.mmax = mmax
        self.lmax = lmax
        self.num_channel_in = num_channel_in
        self.num_channel_out = num_channel_out
        self.weight_type = weight_type

        if num_components_in is None:
            self.num_components_in = [lmax + 1 - m for m in range(mmax + 1)]
        else:
            self.num_components_in = num_components_in
        if num_components_out is None:
            self.num_components_out = [lmax + 1 - m for m in range(mmax + 1)]
        else:
            self.num_components_out = num_components_out

        self.m0_rlinear = Linear(
            self.num_channel_in * self.num_components_in[0],
            self.num_channel_out * self.num_components_out[0],
            bias=True,
        )
        self.ms_clinear = nn.ModuleList()
        for m in range(1, self.mmax + 1):
            self.ms_clinear.append(
                uvSO2MLinear(
                    m,
                    self.num_channel_in,
                    self.num_channel_out,
                    self.num_components_in[m],
                    self.num_components_out[m],
                    weight_type=weight_type,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        Cout = self.num_channel_out
        outputs = []

        xm0 = x.narrow(1, 0, self.num_components_in[0])
        xm0 = xm0.reshape(B, self.num_components_in[0] * self.num_channel_in)
        xm0 = self.m0_rlinear(xm0)
        xm0 = xm0.view(B, self.num_components_out[0], Cout)
        outputs.append(xm0)

        offset = self.lmax + 1
        for m in range(1, self.mmax + 1):
            xm = x.narrow(1, offset, 2 * self.num_components_in[m])
            offset = offset + 2 * self.num_components_in[m]
            xm = xm.reshape(B, 2, self.num_components_in[m] * self.num_channel_in)
            xm = self.ms_clinear[m - 1](xm, concat_outputs=False)
            xr, xi = xm[0], xm[1]
            outputs.append(xr.view(B, self.num_components_out[m], Cout))
            outputs.append(xi.view(B, self.num_components_out[m], Cout))
        return torch.cat(outputs, dim=1)
