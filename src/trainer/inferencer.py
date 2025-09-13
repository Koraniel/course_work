import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                if not met.datasetwise:
                    metrics.update(met.name, met(**batch))

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["logits"].shape[0]
        current_id = batch_idx * batch_size

        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            logits = batch["logits"][i].clone()
            label = batch["labels"][i].clone()
            pred_label = logits.argmax(dim=-1)

            output_id = current_id + i

            output = {
                "pred_label": pred_label,
                "label": label,
            }

            if self.save_path is not None:
                # you can use safetensors or other lib here
                torch.save(output, self.save_path / part / f"output_{output_id}.pth")

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():

            # This is surley disgusting. I feel bad writing this.
            # This will not work in general case but for me it is enough for now.
            is_batchwise_metric = False
            for met in self.metrics["inference"]:
                if not met.datasetwise:
                    is_batchwise_metric = True
                    break
            if is_batchwise_metric:
                for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
                ):
                    batch = self.process_batch(
                        batch_idx=batch_idx,
                        batch=batch,
                        part=part,
                        metrics=self.evaluation_metrics,
                    )
            results = self.process_dataset(part, dataloader)
            self.calculate_metrics(results, self.evaluation_metrics)
        return self.evaluation_metrics.result()
    
    def calculate_metrics(self, dataset_results, metrics: MetricTracker):
        metric_funcs = self.metrics["inference"]

        for met in metric_funcs:
            if met.datasetwise:
                metrics.update(met.name, met(**dataset_results))

    def process_dataset(self, part, dataloader):
        self.model.eval()
        self.train = False
        results = {"logits": None, "labels": None, "filenames": []}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=part, total=len(dataloader)):
                results["filenames"].extend(batch["filenames"])
                if results["labels"] is None:
                    results["labels"] = batch["labels"]
                else:
                    results["labels"] = torch.cat((results["labels"], batch["labels"]))
                batch = self.move_batch_to_device(batch)
                batch = self.transform_batch(batch)  # transform batch on device -- faster
                outputs = self.model(**batch)
                sr = 16000
                A = 3/50
                B = 7/50
                s = 15.0
                normW = F.normalize(outputs['W'])
                normv = F.normalize(outputs['vectors'])
                logits = normv @ normW
                one_hot = F.one_hot(batch["labels"], num_classes=logits.shape[1]).float()
                m = (batch['duration'] / sr) * A + B
                logits = s * (logits - m * one_hot)
                if results["logits"] is None:
                    results["logits"] = logits
                else:
                    results["logits"] = torch.cat((results["logits"], logits))
        results['part'] = part
        return results