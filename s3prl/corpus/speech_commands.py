import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple, Union

import torchaudio

from s3prl import Container, cache
from s3prl.base.output import Output
from s3prl.util import registry

from .base import Corpus

torchaudio.set_audio_backend("sox_io")
logger = logging.getLogger(__name__)

CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "_unknown_",
    "_silence_",
]


class SpeechCommandsV1(Corpus):
    """
    Args:
        dataset_root (str): should contain a 'dev' sub-folder for the training/validation set
            and a 'test' sub-folder for the testing set
    """

    def __init__(self, dataset_root: str, n_jobs: int = 4) -> None:
        dataset_root = Path(dataset_root)
        train_dataset_root = dataset_root / "dev"
        test_dataset_root = dataset_root / "test"

        train_list, valid_list = self.split_dataset(train_dataset_root)
        train_list = self.parse_train_valid_data_list(train_list, train_dataset_root)
        valid_list = self.parse_train_valid_data_list(valid_list, train_dataset_root)
        test_list = self.parse_test_data_list(test_dataset_root)

        self.train = self.list_to_dict(train_list)
        self.valid = self.list_to_dict(valid_list)
        self.test = self.list_to_dict(test_list)

        self._data = Container()
        self._data.add(self.train)
        self._data.update(self.valid, override=True)  # background noises are duplicated
        self._data.add(self.test)

    @staticmethod
    @cache()
    def split_dataset(
        root_dir: Union[str, Path], max_uttr_per_class=2**27 - 1
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Split Speech Commands into 3 set.

        Args:
            root_dir: speech commands dataset root dir
            max_uttr_per_class: predefined value in the original paper

        Return:
            train_list: [(class_name, audio_path), ...]
            valid_list: as above
        """
        train_list, valid_list = [], []

        for entry in Path(root_dir).iterdir():
            if not entry.is_dir() or entry.name == "_background_noise_":
                continue

            for audio_path in entry.glob("*.wav"):
                speaker_hashed = re.sub(r"_nohash_.*$", "", audio_path.name)
                hashed_again = hashlib.sha1(speaker_hashed.encode("utf-8")).hexdigest()
                percentage_hash = (int(hashed_again, 16) % (max_uttr_per_class + 1)) * (
                    100.0 / max_uttr_per_class
                )

                if percentage_hash < 10:
                    valid_list.append((entry.name, audio_path))
                elif percentage_hash < 20:
                    pass  # testing set is discarded
                else:
                    train_list.append((entry.name, audio_path))

        return train_list, valid_list

    @staticmethod
    @cache()
    def parse_train_valid_data_list(data_list, train_dataset_root: Path):
        data = [
            (class_name, audio_path)
            if class_name in CLASSES
            else ("_unknown_", audio_path)
            for class_name, audio_path in data_list
        ]
        data += [
            ("_silence_", audio_path)
            for audio_path in Path(train_dataset_root, "_background_noise_").glob(
                "*.wav"
            )
        ]
        return data

    @staticmethod
    @cache()
    def parse_test_data_list(test_dataset_root: Path):
        data = [
            (class_dir.name, audio_path)
            for class_dir in Path(test_dataset_root).iterdir()
            if class_dir.is_dir()
            for audio_path in class_dir.glob("*.wav")
        ]
        return data

    @staticmethod
    def path_to_unique_name(path: str):
        return "/".join(Path(path).parts[-2:])

    @classmethod
    def list_to_dict(cls, data_list):
        data = Container(
            {
                cls.path_to_unique_name(audio_path): {
                    "wav_path": audio_path,
                    "class_name": class_name,
                }
                for class_name, audio_path in data_list
            }
        )
        return data

    @property
    def all_data(self):
        """
        Return:
            Container: id (str)
                wav_path (str)
                class_name (str)
        """
        return self._data

    @property
    def data_split_ids(self):
        return list(self.train.keys()), list(self.valid.keys()), list(self.test.keys())

    @classmethod
    def download_dataset(cls, tgt_dir: str) -> None:
        import os
        import tarfile

        import requests

        assert os.path.exists(
            os.path.abspath(tgt_dir)
        ), "Target directory does not exist"

        def unzip_targz_then_delete(filepath: str, filename: str):
            file_path = os.path.join(
                os.path.abspath(tgt_dir), "CORPORA_DIR", filename.replace(".tar.gz", "")
            )
            os.makedirs(file_path)

            with tarfile.open(os.path.abspath(filepath)) as tar:
                tar.extractall(path=os.path.abspath(file_path))
            os.remove(os.path.abspath(filepath))

        def download_from_url(url: str):
            filename = url.split("/")[-1].replace(" ", "_")
            filepath = os.path.join(tgt_dir, filename)

            r = requests.get(url, stream=True)
            if r.ok:
                logger.info(f"Saving {filename} to", os.path.abspath(filepath))
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024 * 10):
                        if chunk:
                            f.write(chunk)
                            f.flush()
                            os.fsync(f.fileno())
                logger.info(f"{filename} successfully downloaded")
                unzip_targz_then_delete(filepath, filename)
            else:
                logger.info(f"Download failed: status code {r.status_code}\n{r.text}")

        if not os.path.exists(
            os.path.join(os.path.abspath(tgt_dir), "CORPORA_DIR/speech_commands_v0.01")
        ):
            download_from_url(
                "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
            )
        if not os.path.exists(
            os.path.join(
                os.path.abspath(tgt_dir), "CORPORA_DIR/speech_commands_test_set_v0.01"
            )
        ):
            download_from_url(
                "http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz"
            )
        logger.info(
            f"Speech commands dataset downloaded. Located at {os.path.abspath(tgt_dir)}/CORPORA_DIR/"
        )


@registry.put()
def gsc_v1_for_superb(dataset_root: str, n_jobs: int = 4):
    corpus = SpeechCommandsV1(dataset_root, n_jobs)

    def format_fields(data: dict):
        formated_data = Container()
        for key, value in data.items():
            data_point = {
                "wav_path": value.wav_path,
                "label": value.class_name,
                "start_sec": None,
                "end_sec": None,
            }
            if value.class_name == "_silence_":
                info = torchaudio.info(value.wav_path)
                for start in list(range(info.num_frames))[::info.sample_rate]:
                    seg = data_point.copy()
                    end = min(start + 1 * info.sample_rate, info.num_frames)
                    seg["start_sec"] = start / info.sample_rate
                    seg["end_sec"] = end / info.sample_rate
                    formated_data[f"{key}_{start}_{end}"] = seg
            else:
                formated_data[key] = data_point
        return formated_data

    train_data, valid_data, test_data = corpus.data_split
    return Output(
        train_data=format_fields(train_data),
        valid_data=format_fields(valid_data),
        test_data=format_fields(test_data),
    )
