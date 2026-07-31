import torch
import torch.nn as nn
import math

PI = math.pi
sqrt = math.sqrt


class RealSphericalHarmonics(nn.Module):
    def __init__(self, degrees: list[int]):
        super().__init__()
        max_l = max(degrees)
        assert 0 <= max_l <= 4, "This implementation supports l_max in [0, 4]"
        self.degrees = degrees

    def forward(self, vecs: torch.Tensor) -> torch.Tensor:
        assert vecs.shape[-1] == 3, "Input must have shape [batch, 3]"
        x, y, z = torch.unbind(
            torch.nn.functional.normalize(vecs, dim=-1), dim=-1
        )
        n = vecs.shape[0]
        total = sum(2 * l + 1 for l in self.degrees)
        out = torch.empty(n, total, dtype=vecs.dtype, device=vecs.device)
        idx = 0
        for degree in self.degrees:
            # l = 0
            if degree == 0:
                out[:, idx] = 0.5 * sqrt(1 / PI)
                idx += 1

            # l = 1
            if degree == 1:
                c1 = sqrt(3 / (4 * PI))
                out[:, idx] = c1 * y  # Y_1^-1
                out[:, idx + 1] = c1 * z  # Y_10
                out[:, idx + 2] = c1 * x  # Y_11
                idx += 3

            # l = 2
            if degree == 2:
                c2a = 0.5 * sqrt(15 / PI)
                c2b = 0.25 * sqrt(5 / PI)
                c2c = 0.25 * sqrt(15 / PI)
                out[:, idx] = c2a * x * y  # Y_2^-2
                out[:, idx + 1] = c2a * y * z  # Y_2^-1
                out[:, idx + 2] = c2b * (3 * z**2 - 1)  # Y_20
                out[:, idx + 3] = c2a * x * z  # Y_21
                out[:, idx + 4] = c2c * (x**2 - y**2)  # Y_22
                idx += 5

            # l = 3
            if degree == 3:
                c3a = 0.25 * sqrt(35 / (2 * PI))
                c3b = 0.5 * sqrt(105 / PI)
                c3c = 0.25 * sqrt(21 / (2 * PI))
                c3d = 0.25 * sqrt(7 / PI)
                c3e = 0.25 * sqrt(105 / PI)
                out[:, idx] = c3a * y * (3 * x**2 - y**2)  # Y_3^-3
                out[:, idx + 1] = c3b * x * y * z  # Y_3^-2
                out[:, idx + 2] = c3c * y * (5 * z**2 - 1)  # Y_3^-1
                out[:, idx + 3] = c3d * (5 * z**3 - 3 * z)  # Y_30
                out[:, idx + 4] = c3c * x * (5 * z**2 - 1)  # Y_31
                out[:, idx + 5] = c3e * (x**2 - y**2) * z  # Y_32
                out[:, idx + 6] = c3a * x * (x**2 - 3 * y**2)  # Y_33
                idx += 7

            # l = 4
            if degree == 4:
                c4a = 0.75 * sqrt(35 / PI)
                c4b = 0.75 * sqrt(35 / (2 * PI))
                c4c = 0.75 * sqrt(5 / PI)
                c4d = 0.75 * sqrt(5 / (2 * PI))
                c4e = 0.1875 * sqrt(1 / PI)
                c4f = 0.375 * sqrt(5 / PI)
                c4g = 0.1875 * sqrt(35 / PI)
                out[:, idx] = c4a * x * y * (x**2 - y**2)  # Y_4^-4
                out[:, idx + 1] = c4b * y * (3 * x**2 - y**2) * z  # Y_4^-3
                out[:, idx + 2] = c4c * x * y * (7 * z**2 - 1)  # Y_4^-2
                out[:, idx + 3] = c4d * y * (7 * z**3 - 3 * z)  # Y_4^-1
                out[:, idx + 4] = c4e * (35 * z**4 - 30 * z**2 + 3)  # Y_40
                out[:, idx + 5] = c4d * x * (7 * z**3 - 3 * z)  # Y_41
                out[:, idx + 6] = (
                    c4f * (x**2 - y**2) * (7 * z**2 - 1)
                )  # Y_42
                out[:, idx + 7] = c4b * x * (x**2 - 3 * y**2) * z  # Y_43
                out[:, idx + 8] = c4g * (
                    x**2 * (x**2 - 3 * y**2)
                    - y**2 * (3 * x**2 - y**2)
                )  # Y_44
                idx += 9

        return out
