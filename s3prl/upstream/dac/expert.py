from collections import OrderedDict
from typing import Dict, List, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch
import torchaudio
import sys

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        import dac

        if ckpt is None:
            ckpt = dac.utils.download(model_type="16khz")
        self.model = dac.DAC.load(ckpt)
        self.model.eval()
        self.model.to(device)

    def get_downsample_rates(self, key: str) -> int:
        return 16000 / 50

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        wavs = pad_sequence(wavs, batch_first=True).unsqueeze(1)
        # wavs: (batch_size, 1, max_len)
        x = self.model.preprocess(wavs, 16000)
        z, codes, latents, _, _ = self.model.encode(x, n_quantizers=3)
        token_embeddings = z.permute((0, 2, 1))    # [B, D, T] -> [B, T, D]

        # Sainity check
        # y = self.model.decode(z)
        # torchaudio.save("test.wav", y[0].cpu(), 16000)

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": token_embeddings,
            # "PR": [hidden, feature],
            # "ASR": [hidden, feature],
            # "QbE": [hidden, feature],
            # "SID": [hidden, feature],
            # "ASV": [hidden, feature],
            # "SD": [hidden, feature],
            # "ER": [hidden, feature],
            # "SF": [hidden, feature],
            # "SE": [hidden, feature],
            # "SS": [hidden, feature],
            # "secret": [hidden, feature],
        }
