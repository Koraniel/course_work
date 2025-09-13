import torch
import random

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["data_object"] = torch.vstack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])
    result_batch["filenames"] = [elem["filename"] for elem in dataset_items]

    return result_batch

def collate_fn_aasist(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["data_object"] = torch.vstack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])
    result_batch["filenames"] = [elem["filename"] for elem in dataset_items]

    return result_batch

def collate_fn_aasist2_train(dataset_items: list[dict]):
    def change_len(x, L):
        while x.shape[0] < L:
            x = torch.cat((x, x), dim=0)
        return x[:L]

    N = random.uniform(1.0, 6.0)
    L = int(N * 16000)

    result_batch = {}

    # example of collate_fn
    result_batch["data_object"] = torch.vstack(
        [change_len(elem["data_object"], L) for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])
    result_batch["filenames"] = [elem["filename"] for elem in dataset_items]
    result_batch["duration"] = L
    return result_batch

def collate_fn_aasist2_not_train(dataset_items: list[dict]):
    def change_len(x, L):
        while x.shape[0] < L:
            x = torch.cat((x, x), dim=0)
        return x[:L]
    N = 6.0
    L = int(N * 16000)

    result_batch = {}

    result_batch["data_object"] = torch.vstack(
        [change_len(elem["data_object"], L) for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])
    result_batch["filenames"] = [elem["filename"] for elem in dataset_items]
    result_batch["duration"] = L
    return result_batch

collate_fn_dict = {"aasist": {
                       "train": collate_fn_aasist,
                       "dev": collate_fn_aasist,
                       "eval": collate_fn_aasist
                   },
                   "aasist2": {
                       "train": collate_fn_aasist2_train,
                       "dev": collate_fn_aasist2_not_train,
                       "eval": collate_fn_aasist2_not_train
                   }
                   }