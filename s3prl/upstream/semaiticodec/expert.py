from collections import OrderedDict
import math
from typing import Dict, List, Union

import torch.nn as nn
from torch import Tensor
import torch
import torchaudio
import sys
from .codec_wrapper import CodecWrapper


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
        self.model = CodecWrapper()

    def get_downsample_rates(self, key: str) -> int:
        return 16000 / 100 * 2

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        token_embeddings = self.model.forward(wavs) # codes: (B, T, D)

        orig_len = max([len(wav) for wav in wavs])
        new_len = math.ceil(orig_len // self.get_downsample_rates("ASR"))
        assert token_embeddings.size(1) >= new_len - 1, f"{token_embeddings.shape} vs {new_len}"
        token_embeddings = token_embeddings[:, :new_len, :]

        # Sainity check
        # audio_out = self.model.decoder(token_embeddings)
        # audio_path = "/mnt/users/hccl.local/jkzhao/projects/s3prl/debug.wav"
        # torchaudio.save(audio_path, audio_out.cpu()[0], sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
        # import pdb; pdb.set_trace()

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
