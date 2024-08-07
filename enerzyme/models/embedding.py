import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ELECTRON_CONFIG = np.array([
  #  Z 1s 2s 2p 3s 3p 4s  3d 4p 5s  4d 5p 6s  4f  5d 6p vs vp  vd  vf
  [  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0], # n
  [  1, 1, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # H
  [  2, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # He
  [  3, 2, 1, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # Li
  [  4, 2, 2, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Be
  [  5, 2, 2, 1, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 1,  0,  0], # B
  [  6, 2, 2, 2, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 2,  0,  0], # C
  [  7, 2, 2, 3, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 3,  0,  0], # N
  [  8, 2, 2, 4, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 4,  0,  0], # O
  [  9, 2, 2, 5, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 5,  0,  0], # F
  [ 10, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0], # Ne
  [ 11, 2, 2, 6, 1, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # Na
  [ 12, 2, 2, 6, 2, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Mg
  [ 13, 2, 2, 6, 2, 1, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 1,  0,  0], # Al
  [ 14, 2, 2, 6, 2, 2, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 2,  0,  0], # Si
  [ 15, 2, 2, 6, 2, 3, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 3,  0,  0], # P
  [ 16, 2, 2, 6, 2, 4, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 4,  0,  0], # S
  [ 17, 2, 2, 6, 2, 5, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 5,  0,  0], # Cl
  [ 18, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0], # Ar
  [ 19, 2, 2, 6, 2, 6, 1,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # K
  [ 20, 2, 2, 6, 2, 6, 2,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Ca
  [ 21, 2, 2, 6, 2, 6, 2,  1, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  1,  0], # Sc
  [ 22, 2, 2, 6, 2, 6, 2,  2, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  2,  0], # Ti
  [ 23, 2, 2, 6, 2, 6, 2,  3, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  3,  0], # V
  [ 24, 2, 2, 6, 2, 6, 1,  5, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  5,  0], # Cr
  [ 25, 2, 2, 6, 2, 6, 2,  5, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  5,  0], # Mn
  [ 26, 2, 2, 6, 2, 6, 2,  6, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  6,  0], # Fe
  [ 27, 2, 2, 6, 2, 6, 2,  7, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  7,  0], # Co
  [ 28, 2, 2, 6, 2, 6, 2,  8, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  8,  0], # Ni
  [ 29, 2, 2, 6, 2, 6, 1, 10, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0, 10,  0], # Cu
  [ 30, 2, 2, 6, 2, 6, 2, 10, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0, 10,  0], # Zn
  [ 31, 2, 2, 6, 2, 6, 2, 10, 1, 0,  0, 0, 0,  0,  0, 0, 2, 1, 10,  0], # Ga
  [ 32, 2, 2, 6, 2, 6, 2, 10, 2, 0,  0, 0, 0,  0,  0, 0, 2, 2, 10,  0], # Ge
  [ 33, 2, 2, 6, 2, 6, 2, 10, 3, 0,  0, 0, 0,  0,  0, 0, 2, 3, 10,  0], # As
  [ 34, 2, 2, 6, 2, 6, 2, 10, 4, 0,  0, 0, 0,  0,  0, 0, 2, 4, 10,  0], # Se
  [ 35, 2, 2, 6, 2, 6, 2, 10, 5, 0,  0, 0, 0,  0,  0, 0, 2, 5, 10,  0], # Br
  [ 36, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 2, 6, 10,  0], # Kr
  [ 37, 2, 2, 6, 2, 6, 2, 10, 6, 1,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # Rb
  [ 38, 2, 2, 6, 2, 6, 2, 10, 6, 2,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Sr
  [ 39, 2, 2, 6, 2, 6, 2, 10, 6, 2,  1, 0, 0,  0,  0, 0, 2, 0,  1,  0], # Y
  [ 40, 2, 2, 6, 2, 6, 2, 10, 6, 2,  2, 0, 0,  0,  0, 0, 2, 0,  2,  0], # Zr
  [ 41, 2, 2, 6, 2, 6, 2, 10, 6, 1,  4, 0, 0,  0,  0, 0, 1, 0,  4,  0], # Nb
  [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 1,  5, 0, 0,  0,  0, 0, 1, 0,  5,  0], # Mo
  [ 43, 2, 2, 6, 2, 6, 2, 10, 6, 2,  5, 0, 0,  0,  0, 0, 2, 0,  5,  0], # Tc
  [ 44, 2, 2, 6, 2, 6, 2, 10, 6, 1,  7, 0, 0,  0,  0, 0, 1, 0,  7,  0], # Ru
  [ 45, 2, 2, 6, 2, 6, 2, 10, 6, 1,  8, 0, 0,  0,  0, 0, 1, 0,  8,  0], # Rh
  [ 46, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0], # Pd
  [ 47, 2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0,  0,  0, 0, 1, 0, 10,  0], # Ag
  [ 48, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 2, 0, 10,  0], # Cd
  [ 49, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0,  0,  0, 0, 2, 1, 10,  0], # In
  [ 50, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0,  0,  0, 0, 2, 2, 10,  0], # Sn
  [ 51, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0,  0,  0, 0, 2, 3, 10,  0], # Sb
  [ 52, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0,  0,  0, 0, 2, 4, 10,  0], # Te
  [ 53, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0,  0,  0, 0, 2, 5, 10,  0], # I
  [ 54, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0,  0,  0, 0, 2, 6, 10,  0], # Xe
  [ 55, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1,  0,  0, 0, 1, 0,  0,  0], # Cs
  [ 56, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  0,  0, 0, 2, 0,  0,  0], # Ba
  [ 57, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  0,  1, 0, 2, 0,  1,  0], # La
  [ 58, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  1,  1, 0, 2, 0,  1,  1], # Ce
  [ 59, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  3,  0, 0, 2, 0,  0,  3], # Pr
  [ 60, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  4,  0, 0, 2, 0,  0,  4], # Nd
  [ 61, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  5,  0, 0, 2, 0,  0,  5], # Pm
  [ 62, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  6,  0, 0, 2, 0,  0,  6], # Sm
  [ 63, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  7,  0, 0, 2, 0,  0,  7], # Eu
  [ 64, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  7,  1, 0, 2, 0,  1,  7], # Gd
  [ 65, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  9,  0, 0, 2, 0,  0,  9], # Tb
  [ 66, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 10,  0, 0, 2, 0,  0, 10], # Dy
  [ 67, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 11,  0, 0, 2, 0,  0, 11], # Ho
  [ 68, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 12,  0, 0, 2, 0,  0, 12], # Er
  [ 69, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 13,  0, 0, 2, 0,  0, 13], # Tm
  [ 70, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  0, 0, 2, 0,  0, 14], # Yb
  [ 71, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  1, 0, 2, 0,  1, 14], # Lu
  [ 72, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  2, 0, 2, 0,  2, 14], # Hf
  [ 73, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  3, 0, 2, 0,  3, 14], # Ta
  [ 74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  4, 0, 2, 0,  4, 14], # W
  [ 75, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  5, 0, 2, 0,  5, 14], # Re
  [ 76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  6, 0, 2, 0,  6, 14], # Os
  [ 77, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  7, 0, 2, 0,  7, 14], # Ir
  [ 78, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14,  9, 0, 1, 0,  9, 14], # Pt
  [ 79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0, 1, 0, 10, 14], # Au
  [ 80, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 2, 0, 10, 14], # Hg
  [ 81, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1, 2, 1, 10, 14], # Tl
  [ 82, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2, 2, 2, 10, 14], # Pb
  [ 83, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3, 2, 3, 10, 14], # Bi
  [ 84, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 4, 2, 4, 10, 14], # Po
  [ 85, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 5, 2, 5, 10, 14], # At
  [ 86, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 10, 14]  # Rn
], dtype=np.float64)
# normalize entries (between 0.0 and 1.0)
ELECTRON_CONFIG = ELECTRON_CONFIG / np.max(ELECTRON_CONFIG, axis=0)

class NuclearEmbedding(nn.Module):
    """
    Embedding which maps scalar nuclear charges Z to vectors in a
    (num_features)-dimensional feature space. The embedding consists of a freely
    learnable parameter matrix [Zmax, num_features] and a learned linear mapping
    from the electron configuration to a (num_features)-dimensional vector. The
    latter part encourages alchemically meaningful representations without
    restricting the expressivity of learned embeddings.

    Params:
    -----
    num_features: Dimensions of feature space.
    
    Zmax: Maximum nuclear charge +1 of atoms. The default is 87, so all
        elements up to Rn (Z=86) are supported. Can be kept at the default
        value (has minimal memory impact).

    zero_init: If true, the parameters will be initialized as zero tensors.

    dtype: Data type of floating number parameters.

    use_electron_config: If true, electron configurations will be used in the embedding.
    """

    def __init__(
        self, num_features: int, Zmax: int=87, zero_init: bool=True, dtype: torch.dtype=torch.float64, use_electron_config: bool=True
    ) -> None:
        super.__init__()
        self.num_features = num_features
        self.use_electron_config = use_electron_config
        self.register_parameter(
            "element_embedding", nn.Parameter(torch.Tensor(Zmax, self.num_features, dtype=dtype))
        )
        self.register_buffer(
            "embedding", torch.Tensor(Zmax, self.num_features, dtype=dtype), persistent=False
        )
        if use_electron_config:
            self.register_buffer("electron_config", torch.tensor(ELECTRON_CONFIG, dtype=dtype))
            self.config_linear = nn.Linear(
                self.electron_config.size(1), self.num_features, bias=False, dtype=dtype
            )
        self.reset_parameters(zero_init)

    def reset_parameters(self, zero_init: bool = True) -> None:
        """ Initialize parameters. """
        if zero_init:
            nn.init.zeros_(self.element_embedding)
            if self.use_electron_config:
                nn.init.zeros_(self.config_linear.weight)
        else:
            nn.init.uniform_(self.element_embedding, -math.sqrt(3), math.sqrt(3))
            if self.use_electron_config:
                nn.init.orthogonal_(self.config_linear.weight)

    def train(self, mode: bool = True) -> None:
        """ Switch between training and evaluation mode. """
        super(NuclearEmbedding, self).train(mode=mode)
        if not self.training:
            with torch.no_grad():
                if self.use_electron_config:
                    self.embedding = self.element_embedding + self.config_linear(
                        self.electron_config
                    )
                else:
                    self.embedding = self.element_embedding

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Assign corresponding embeddings to nuclear charges.
        N: Number of atoms.
        num_features: Dimensions of feature space.

        Arguments:
            Z (LongTensor [N]):
                Nuclear charges (atomic numbers) of atoms.

        Returns:
            x (FloatTensor [N, num_features]):
                Embeddings of all atoms.
        """
        if self.training:  # during training, the embedding needs to be recomputed
            if self.use_electron_config:
                self.embedding = self.element_embedding + self.config_linear(
                    self.electron_config
                )
            else:
                self.embedding = self.element_embedding
        if self.embedding.device.type == "cpu":  # indexing is faster on CPUs
            return self.embedding[Z]
        else:  # gathering is faster on GPUs
            return torch.gather(
                self.embedding, 0, Z.view(-1, 1).expand(-1, self.num_features)
            )
