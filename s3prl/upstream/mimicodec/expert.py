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
        if "/mnt/users/hccl.local/jkzhao/projects/moshi/moshi" not in sys.path:
            sys.path.append("/mnt/users/hccl.local/jkzhao/projects/moshi/moshi")
        from moshi.models import loaders, MimiModel

        if ckpt is None:
            ckpt = "/mnt/users/hccl.local/jkzhao/projects/moshi/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors"
        self.model = loaders.get_mimi(ckpt, device)
        # self.model.set_num_codebooks(1) # semantic only

        self.in_resampler = torchaudio.transforms.Resample(16000, 24000)
        self.out_resampler = torchaudio.transforms.Resample(24000, 16000)

    def get_downsample_rates(self, key: str) -> int:
        return 16000 / 50

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        wavs = pad_sequence(wavs, batch_first=True).unsqueeze(1)
        # wavs: (batch_size, 1, max_len)
        wavs = self.in_resampler(wavs)
        token_embeddings = self.model.encode_to_latent(wavs) # (B, D, T)
        # token_embeddings_ = self.model._to_encoder_framerate(token_embeddings)   # 2x upsample with conv
        # new_embedding = torch.zeros(token_embeddings_.shape[0], token_embeddings_.shape[1], token_embeddings_.shape[2]*2).to(token_embeddings_.device)
        # new_embedding[:, :, ::2] = token_embeddings_
        # new_embedding[:, :, 1::2] = token_embeddings_

        new_embedding = torch.zeros(token_embeddings.shape[0], token_embeddings.shape[1], token_embeddings.shape[2]*4).to(token_embeddings.device)
        new_embedding[:, :, ::4] = token_embeddings
        new_embedding[:, :, 1::4] = token_embeddings
        new_embedding[:, :, 2::4] = token_embeddings
        new_embedding[:, :, 3::4] = token_embeddings
        token_embeddings_ = new_embedding.permute(0, 2, 1)  # [B, T, D]

        # # Sainity check
        # import pdb; pdb.set_trace()
        # state = self.model._streaming_state
        # token_embeddings = self.model._to_encoder_framerate(token_embeddings)   # 2x upsample with conv
        # if self.model.decoder_transformer is not None:
        #     if state is None:
        #         (token_embeddings,) = self.model.decoder_transformer(token_embeddings)
        #     else:
        #         assert state.graphed_tr_dec is not None
        #         (token_embeddings,) = state.graphed_tr_dec(token_embeddings)
        # with self.model._context_for_encoder_decoder:
        #     audio_out = self.model.decoder(token_embeddings)    # [B, 1, T]
        # audio_out = self.out_resampler(audio_out)
        # audio_path = "/mnt/users/hccl.local/jkzhao/projects/s3prl/debug1.wav"
        # torchaudio.save(audio_path, audio_out.cpu()[0], sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
        # import pdb; pdb.set_trace()

        # The "hidden_states" key will be used as default in many cases
        # Others keys in this example are presented for SUPERB Challenge
        return {
            "hidden_states": token_embeddings_,
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
