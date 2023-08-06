import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WhisperModel

from s3prl.hub import wavlm_base_plus

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DOWNSAMPLE_RATE = 320
WAVLM_BASE_NUM_LAYER = 13
WAVLM_BASE_HIDDEN_SIZE = 768
WHISPER_BASE_NUM_LAYER = 7
WHISPER_HIDDEN_SIZE = 512


def build_mlp(input_size: int, hidden_size: int, output_size: int):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size),
    )


class UpstreamExpert(torch.nn.Module):
    def __init__(
        self,
        project_model: str,
        project_size: int,
        use_wavlm: bool,
        use_whisper: bool,
        extract_noisy_mel: bool,
        extract_clean_mel: bool,
        pretrained_clean_whisper_projects: str,
        whisper_mel: str = "noisy",
        clean_latent_guidance: float = 1.0,
        **kwds,
    ):
        super().__init__()
        assert use_wavlm or use_whisper
        self.whisper_mel = whisper_mel
        self.extract_noisy_mel = extract_noisy_mel
        self.extract_clean_mel = extract_clean_mel
        logger.info("Setup Whisper Mel extractor")
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "openai/whisper-base"
        )

        if project_model == "MLP":
            build_model = partial(
                build_mlp, hidden_size=project_size, output_size=project_size
            )
        else:
            raise ValueError

        self.use_wavlm = use_wavlm
        if use_wavlm:
            logger.info("Setup WavLM")
            self.wavlm = wavlm_base_plus()
            self.wavlm_projects = torch.nn.ModuleList(
                [
                    build_model(WAVLM_BASE_HIDDEN_SIZE)
                    for _ in range(WAVLM_BASE_NUM_LAYER)
                ]
            )

        self.use_whisper = use_whisper
        if use_whisper:
            logger.info("Setup Whisper")
            self.whisper_model = WhisperModel.from_pretrained("openai/whisper-base")
            self.whisper_projects = torch.nn.ModuleList(
                [
                    build_model(WHISPER_HIDDEN_SIZE)
                    for _ in range(WHISPER_BASE_NUM_LAYER)
                ]
            )

        self.clean_latent_guidance = clean_latent_guidance
        if clean_latent_guidance > 0:
            self.clean_whisper_projects = torch.nn.ModuleList(
                [
                    build_model(WHISPER_HIDDEN_SIZE)
                    for _ in range(WHISPER_BASE_NUM_LAYER)
                ]
            )
            self.clean_latent_guidance_loss = torch.nn.L1Loss()
            pretrained_ckpt = torch.load(
                pretrained_clean_whisper_projects, map_location="cpu"
            )
            pretrained_weights = OrderedDict()
            for key, value in pretrained_ckpt["Upstream"].items():
                if key.startswith("whisper_projects"):
                    key = key.replace("whisper_projects", "clean_whisper_projects")
                    pretrained_weights[key] = value
            self.load_state_dict(pretrained_weights, strict=False)
            self.clean_whisper_projects.requires_grad_(False)

        self.register_buffer("device_detector", torch.zeros(1))

    def get_downsample_rates(self, key: str) -> int:
        return DOWNSAMPLE_RATE

    def forward(self, all_upstream_inputs):
        device = self.device_detector.device

        if isinstance(all_upstream_inputs, (list, tuple)):
            wavs = [wav.to(device) for wav in all_upstream_inputs]
            noisy_wavs = wavs
            clean_wavs = wavs
            noisy_mels = self.whisper_feature_extractor(
                [wav.cpu().numpy() for wav in noisy_wavs],
                return_tensors="pt",
                sampling_rate=SAMPLE_RATE,
                do_normalize=True,
            ).input_features.to(device)
            clean_mels = self.whisper_feature_extractor(
                [wav.cpu().numpy() for wav in clean_wavs],
                return_tensors="pt",
                sampling_rate=SAMPLE_RATE,
                do_normalize=True,
            ).input_features.to(device)
        else:

            def to_device(target, device):
                if target is None:
                    return None

                if isinstance(target, (list, tuple)):
                    return [torch.FloatTensor(t).to(device) for t in target]
                else:
                    return torch.FloatTensor(target).to(device)

            noisy_wavs = to_device(all_upstream_inputs["noisy_wavs"], device)
            clean_wavs = to_device(all_upstream_inputs["clean_wavs"], device)
            noisy_mels = to_device(all_upstream_inputs["noisy_mels"], device)
            clean_mels = to_device(all_upstream_inputs["clean_mels"], device)

            if self.extract_noisy_mel:
                noisy_mels = self.whisper_feature_extractor(
                    all_upstream_inputs["noisy_wavs"],
                    return_tensors="pt",
                    sampling_rate=SAMPLE_RATE,
                    do_normalize=True,
                ).input_features

            if self.extract_clean_mel:
                clean_mels = self.whisper_feature_extractor(
                    all_upstream_inputs["clean_wavs"],
                    return_tensors="pt",
                    sampling_rate=SAMPLE_RATE,
                    do_normalize=True,
                ).input_features

        wavs_len = [len(wav) for wav in noisy_wavs]
        max_seq_len = len(list(range(0, max(wavs_len), DOWNSAMPLE_RATE)))

        wavlm_projected_hs = []
        if self.use_wavlm:
            with torch.no_grad():
                self.wavlm.eval()
                wavlm_hs = self.wavlm(noisy_wavs)["hidden_states"]

            wavlm_hs = [
                F.layer_norm(hs[:, :max_seq_len, :], hs.shape[-1:]) for hs in wavlm_hs
            ]
            for hs, project in zip(wavlm_hs, self.wavlm_projects):
                projected_hs = project(hs)
                wavlm_projected_hs.append(projected_hs)

        whisper_projected_hs = []
        if self.use_whisper:
            if self.whisper_mel == "noisy":
                whisper_mels = noisy_mels
            elif self.whisper_mel == "clean":
                whisper_mels = clean_mels

            decoder_input_ids = (
                torch.tensor([[1]]) * self.whisper_model.config.decoder_start_token_id
            )
            with torch.no_grad():
                self.whisper_model.eval()
                result = self.whisper_model(
                    whisper_mels.to(device),
                    decoder_input_ids=decoder_input_ids.to(device),
                    output_hidden_states=True,
                )
            whisper_hs = result.encoder_hidden_states
            whisper_hs = [
                F.layer_norm(hs[:, :max_seq_len, :], hs.shape[-1:]) for hs in whisper_hs
            ]
            for hs, project in zip(whisper_hs, self.whisper_projects):
                projected_hs = project(hs)
                whisper_projected_hs.append(projected_hs)

        if self.clean_latent_guidance > 0:
            clean_whisper_projected_hs = []
            with torch.no_grad():
                self.whisper_model.eval()
                result = self.whisper_model(
                    clean_mels.to(device),
                    decoder_input_ids=decoder_input_ids.to(device),
                    output_hidden_states=True,
                )
                clean_whisper_hs = result.encoder_hidden_states
                clean_whisper_hs = [
                    F.layer_norm(hs[:, :max_seq_len, :], hs.shape[-1:])
                    for hs in clean_whisper_hs
                ]
                for hs, project in zip(clean_whisper_hs, self.clean_whisper_projects):
                    projected_hs = project(hs)
                    clean_whisper_projected_hs.append(projected_hs)

            noisy_whisper_hs = torch.stack(whisper_projected_hs, dim=0).mean(dim=0)
            clean_whisper_hs = torch.stack(clean_whisper_projected_hs, dim=0).mean(
                dim=0
            )
            clean_latent_guidance_loss = self.clean_latent_guidance_loss(
                noisy_whisper_hs, clean_whisper_hs
            )
            scaled_clean_latent_guidance_loss = (
                self.clean_latent_guidance * clean_latent_guidance_loss
            )

        if self.use_wavlm and self.use_whisper:
            hidden_states = []
            for wavlm_hs, whisper_hs in zip(wavlm_projected_hs, whisper_projected_hs):
                min_len = min(wavlm_hs.size(1), whisper_hs.size(1))
                wavlm_hs = wavlm_hs[:, :min_len, :]
                whisper_hs = whisper_hs[:, :min_len, :]
                hidden_states = [*hidden_states, wavlm_hs, whisper_hs]
                hidden_state = torch.stack(hidden_states, dim=0).mean(dim=0)

        elif self.use_wavlm:
            hidden_state = torch.stack(wavlm_projected_hs, dim=0).mean(dim=0)

        elif self.use_whisper:
            hidden_state = torch.stack(whisper_projected_hs, dim=0).mean(dim=0)

        results = {
            "hidden_states": [hidden_state],
        }
        if self.clean_latent_guidance > 0:
            results.update(
                {
                    "clean_latent_guidance_loss": clean_latent_guidance_loss,
                    "scaled_clean_latent_guidance_loss": scaled_clean_latent_guidance_loss,
                }
            )

        return results
