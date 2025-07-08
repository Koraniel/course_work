import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from tqdm.auto import tqdm

# TODO: rewrite datasets into one class because there is no difference but a few words
# TODO: remove funs and paper classes after rewriting
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x
  

class Dataset_ASVspoof2019_train(BaseDataset):
    def __init__(self, LA_PATH=None, *args, **kwargs):
        self.cut = 64600
        index_path = ROOT_PATH / "src" / "data" / "aasist" / "train" / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(LA_PATH)

        super().__init__(index, *args, **kwargs)
    
    def _create_index(self, LA_PATH=None):
        DEFAULT_LA_PATH = ROOT_PATH / "src" / "data" / "LA"
        index = []
        if LA_PATH is None and not DEFAULT_LA_PATH.exists():
            raise Exception("There's no default LA and LA_PATH is None")
        if LA_PATH is None:
            LA_PATH = DEFAULT_LA_PATH
        
        with open(LA_PATH / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt", "r") as f:
            l_meta = f.readlines()

        dataset_length = len(l_meta)
        data_path = ROOT_PATH / "src" / "data" / "aasist" / "train"
        data_path.mkdir(exist_ok=True, parents=True)
        print("Creating Train Dataset")
        for i in tqdm(range(dataset_length)):
            line = l_meta[i]
            _, filename, _, _, label = line.strip().split(" ")
            audiofile, _ = sf.read(str(LA_PATH / "ASVspoof2019_LA_train" / f"flac/{filename}.flac"))
            audiofile_pad = pad_random(audiofile, self.cut)
            audiofile_inp = torch.tensor(audiofile_pad, dtype=torch.float)
            filepath = data_path / f"{filename}.pt"
            torch.save(audiofile_inp, filepath)
            bondafide = 1 if label == "bonafide" else 0
            index.append({"path": str(filepath), "label": bondafide})
        write_json(index, str(data_path / "index.json"))
        return index


class Dataset_ASVspoof2019_dev(BaseDataset):
    def __init__(self, LA_PATH=None, *args, **kwargs):
        self.cut = 64600
        index_path = ROOT_PATH / "src" / "data" / "aasist" / "dev" / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(LA_PATH)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, LA_PATH=None):

        DEFAULT_LA_PATH = ROOT_PATH / "src" / "data" / "LA"
        index = []
        if LA_PATH is None and not DEFAULT_LA_PATH.exists():
            raise Exception("There's no default LA and LA_PATH is None")
        if LA_PATH is None:
            LA_PATH = DEFAULT_LA_PATH
        
        with open(LA_PATH / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt", "r") as f:
            l_meta = f.readlines()

        dataset_length = len(l_meta)
        data_path = ROOT_PATH / "src" / "data" / "aasist" / "dev"
        data_path.mkdir(exist_ok=True, parents=True)
        print("Creating Dev Dataset")
        for i in tqdm(range(dataset_length)):
            line = l_meta[i]
            _, filename, _, _, label = line.strip().split(" ")
            audiofile, _ = sf.read(str(LA_PATH / "ASVspoof2019_LA_dev" / f"flac/{filename}.flac"))
            audiofile_pad = pad_random(audiofile, self.cut)
            audiofile_inp = torch.tensor(audiofile_pad, dtype=torch.float)
            filepath = data_path / f"{filename}.pt"
            torch.save(audiofile_inp, filepath)
            bondafide = 1 if label == "bonafide" else 0
            index.append({"path": str(filepath), "label": bondafide})
        write_json(index, str(data_path / "index.json"))
        return index