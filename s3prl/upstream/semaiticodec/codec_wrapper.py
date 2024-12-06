from torch import nn
import torch
from semanticodec import SemantiCodec
from semanticodec.utils import extract_kaldi_fbank_feature
from torch.nn.utils.rnn import pad_sequence

# Constants
SAMPLE_RATE = 16000
SEGMENT_DURATION = 10.24
MEL_TARGET_LENGTH = 1024
AUDIOMAE_PATCH_DURATION = 0.16
SEGMENT_OVERLAP_RATIO = 0.0625

class CodecWrapper:
    def __init__(self):
        self.model = SemantiCodec(token_rate=100, semantic_vocab_size=32768) # 1.40 kbps
        
    def preprocess(self, waveform):
        # waveform: (1, T), 16000Hz
        sr = 16000
        # if stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        # Calculate the original duration
        original_duration = waveform.shape[1] / sr
        # This is to pad the audio to the multiplication of 0.16 seconds so that the original audio can be reconstructed
        original_duration = original_duration + (
            AUDIOMAE_PATCH_DURATION - original_duration % AUDIOMAE_PATCH_DURATION
        )
        # Calculate the token length in theory
        target_token_len = (
            8 * original_duration / AUDIOMAE_PATCH_DURATION / self.model.stack_factor_K
        )
        segment_sample_length = int(SAMPLE_RATE * SEGMENT_DURATION)
        # Pad audio to the multiplication of 10.24 seconds for easier segmentations

        if waveform.shape[1] % segment_sample_length < segment_sample_length:
            waveform = torch.cat(
                [
                    waveform,
                    torch.zeros(
                        1,
                        int(
                            segment_sample_length
                            - waveform.shape[1] % segment_sample_length
                        ),
                        device=waveform.device,
                    ),
                ],
                dim=1,
            )

        mel_target_length = MEL_TARGET_LENGTH * int(
            waveform.shape[1] / segment_sample_length
        )
        # Calculate the mel spectrogram
        mel = extract_kaldi_fbank_feature(
            waveform, sr, target_length=mel_target_length
        )["ta_kaldi_fbank"].unsqueeze(0)
        mel = mel.squeeze(1)    # No use
        assert mel.shape[-1] == 128 and mel.shape[-2] % 1024 == 0
        return mel, target_token_len

    def forward(self, wavs):
        all_latents = []
        for wav in wavs:
            mel, target_token_len = self.preprocess(wav.unsqueeze(0))
            tokens = self.model.encoder(mel.to(self.model.device))
            latent = self.model.encoder.token_to_quantized_feature(tokens)
            all_latents.append(latent.squeeze(0))
        all_latents = pad_sequence(all_latents, batch_first=True)
        return all_latents
        
        # mels = [self.preprocess(wav.unsqueeze(0))[0] for wav in wavs]
        # mels = pad_sequence(mels, batch_first=True)
        # print(mels.shape)
        # # mels: (B, T, D), 16000Hz
        # tokens = self.model.encoder(mels.to(self.model.device))
        # latent = self.model.encoder.token_to_quantized_feature(tokens)
        # return latent
        