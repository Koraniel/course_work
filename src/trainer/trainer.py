from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            if not met.datasetwise:
                metrics.update(met.name, met(**batch))
        return batch
    
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
    
    # TODO: make this work not only for inference
    # Now I understand that I need to completely rewrite structure of 
    # the whole template in order to do this how I want   T_T
    def calculate_metrics(self, dataset_results, metrics: MetricTracker):
        metric_funcs = self.metrics["inference"]

        for met in metric_funcs:
            if met.datasetwise:
                metrics.update(met.name, met(**dataset_results))
        

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
