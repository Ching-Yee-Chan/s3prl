from collections import OrderedDict
from typing import Dict, List, Optional, Union

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
        from transformers import EncodecModel, AutoProcessor

        if ckpt is None:
            ckpt = "facebook/encodec_24khz"
        self.model =  EncodecModel.from_pretrained(ckpt)
        # self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model.eval()
        self.model.to(device)

        self.in_resampler = torchaudio.transforms.Resample(16000, 24000)
        self.out_resampler = torchaudio.transforms.Resample(24000, 16000)

    def get_downsample_rates(self, key: str) -> int:
        return 16000 / 75
    
    def _decode_frame(self, codes: torch.Tensor, scale: Optional[torch.Tensor] = None, to_audio: bool = False) -> torch.Tensor:
        codes = codes.transpose(0, 1)
        embeddings = self.model.quantizer.decode(codes)
        if to_audio:
            outputs = self.model.decoder(embeddings)
            if scale is not None:
                outputs = outputs * scale.view(-1, 1, 1)
            return outputs
        else:
            return embeddings
    
    def decode(
        self,
        audio_codes: torch.Tensor,
        audio_scales: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        to_audio: bool = False,
    ):
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
                Scaling factor for each `audio_codes` input.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        assert self.model.config.chunk_length is None, "embedding overlapping is not supported!"
        if len(audio_codes) != 1:
            raise ValueError(f"Expected one frame, got {len(audio_codes)}")
        audio_values = self._decode_frame(audio_codes[0], audio_scales[0], to_audio)

        # truncate based on padding mask
        if to_audio and padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        return audio_values

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        wavs = [self.in_resampler(wav) for wav in wavs]
        wav_lengths = [wav.size(-1) for wav in wavs]
        input_values = pad_sequence(wavs, batch_first=True).unsqueeze(1)    # [B, 1, T]
        padding_mask = torch.stack([torch.arange(max(wav_lengths)) < wav_len for wav_len in wav_lengths], dim=0).unsqueeze(1).to(input_values.device)   # [B, 1, T]
        # 3.0 -> 4 layers
        encoder_outputs = self.model.encode(input_values, padding_mask, bandwidth=3.0) # codes: (B, D, T)

        sainity_check = False    # Set to True to check the audio reconstruction
        audio_values = self.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, padding_mask, to_audio=sainity_check)
        if sainity_check:
            output_wav = self.out_resampler(audio_values).cpu()[0]
            audio_path = "/mnt/users/hccl.local/jkzhao/projects/s3prl/debug.wav"
            torchaudio.save(audio_path, output_wav, sample_rate=16000, encoding='PCM_S', bits_per_sample=16)
            token_embeddings = torch.zeros(1, 1, output_wav.size(-1)//320, 512).to(input_values.device)
        else:
            token_embeddings = audio_values.permute(0, 2, 1)  # [B, T, D]

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
