import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from tqdm.auto import tqdm


class Dataset_ASVspoof2019_general(BaseDataset):
    def __init__(self, dataset_type: str, LA_PATH=None, *args, **kwargs):
        """
        Args:
            dataset_type (str): "train" or "dev" or "eval"
        """
        if dataset_type not in ["train", "dev", "eval"]:
            raise TypeError("dataset_type must be \"train\" or \"dev\" or \"eval\"")
        self.cut = 64600
        self.dataset_type = dataset_type
        index_path = ROOT_PATH / "src" / "data" / "aasist" / self.dataset_type / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(LA_PATH)
        
        # FOR TESTING
        # if self.dataset_type == "train":
        #     index = index[:10]
        # else:
        #     index = index[2000:3000]

        super().__init__(index, *args, **kwargs)
    
    def _create_index(self, LA_PATH=None):
        DEFAULT_LA_PATH = ROOT_PATH / "src" / "data" / "LA"
        index = []

        if LA_PATH is None and not DEFAULT_LA_PATH.exists():
            raise LookupError("There's no default LA and LA_PATH is None")
        if LA_PATH is None:
            LA_PATH = DEFAULT_LA_PATH
        
        if self.dataset_type == "train":
            with open(LA_PATH / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{self.dataset_type}.trn.txt", "r") as f:
                l_meta = f.readlines()
        else:
            with open(LA_PATH / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{self.dataset_type}.trl.txt", "r") as f:
                l_meta = f.readlines()
        
        dataset_length = len(l_meta)
        data_path = ROOT_PATH / "src" / "data" / "aasist" / self.dataset_type
        data_path.mkdir(exist_ok=True, parents=True)
        print(f"Creating {self.dataset_type} dataset.")

        for i in tqdm(range(dataset_length)):
            line = l_meta[i]
            _, filename, _, _, label = line.strip().split(" ")
            audiofile, _ = sf.read(str(LA_PATH / f"ASVspoof2019_LA_{self.dataset_type}" / f"flac/{filename}.flac"))
            audiofile_pad = self._pad_random(audiofile, self.cut)
            audiofile_inp = torch.tensor(audiofile_pad, dtype=torch.float)
            filepath = data_path / f"{filename}.pt"
            torch.save(audiofile_inp, filepath)
            bondafide = 1 if label == "bonafide" else 0
            index.append({"path": str(filepath), "filename": filename, "label": bondafide})

        write_json(index, str(data_path / "index.json"))
        return index
    
    def _pad_random(self, x: np.ndarray, max_len: int = 64600):
        x_len = x.shape[0]
        # if duration is already long enough
        if x_len > max_len:
            stt = np.random.randint(x_len - max_len)
            return x[stt:stt + max_len]

        # if too short
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (num_repeats))[:max_len]
        return padded_x
