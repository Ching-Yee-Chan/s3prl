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
        if "/mnt/users/hccl.local/jkzhao/projects/WavTokenizer" not in sys.path:
            sys.path.append("/mnt/users/hccl.local/jkzhao/projects/WavTokenizer")
        from decoder.pretrained import WavTokenizer

        if ckpt is None:
            ckpt = "/mnt/users/hccl.local/jkzhao/projects/WavTokenizer/result/pretrain/wavtokenizer_large_speech_320_24k.ckpt"
        if model_config is None:
            model_config = "/mnt/users/hccl.local/jkzhao/projects/WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
        self.model = WavTokenizer.from_pretrained0802(model_config, ckpt)
        self.model.to(device)

        self.in_resampler = torchaudio.transforms.Resample(16000, 24000)
        self.out_resampler = torchaudio.transforms.Resample(24000, 16000)

    def get_downsample_rates(self, key: str) -> int:
        return 16000 / 75

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        wavs = pad_sequence(wavs, batch_first=True)
        # wavs: (batch_size, max_len)
        wavs = self.in_resampler(wavs)
        bandwidth_id = torch.tensor([0])    # No use
        token_embeddings, discrete_code = self.model.encode_infer(wavs, bandwidth_id=bandwidth_id)  # [B, D, T]
        token_embeddings = token_embeddings.permute(0, 2, 1)  # [B, T, D], COMMENT OUT WHEN DOING SANITY CHECK

        # Sainity check
        # self.model.to("cpu")
        # self.out_resampler.to("cpu")
        # audio_out = self.model.decode(token_embeddings.cpu(), bandwidth_id=bandwidth_id.cpu())
        # audio_out = self.out_resampler(audio_out) 
        # audio_path = "/mnt/users/hccl.local/jkzhao/projects/s3prl/debug.wav"
        # torchaudio.save(audio_path, audio_out[:1], sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
        # self.model.to("cuda")
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
