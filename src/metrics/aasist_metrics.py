import torch
import torchmetrics
import numpy as np
from src.metrics.base_metric import BaseMetric
from src.utils.io_utils import ROOT_PATH


class EERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, filenames, *args, **kwargs):
        scores = torch.nn.functional.sigmoid(logits[:, 1] - logits[:, 0]).cpu().detach().numpy()
        argscores = np.argsort(-scores)
        labels = labels.cpu().detach().numpy()[argscores]
        scores = scores[argscores]
        acceptance_count = labels.sum()
        rejection_count = labels.shape[0] - acceptance_count
        false_rejection_ratio_prefix = 1 - np.cumsum(labels) / acceptance_count
        false_acceptance_ratio_suffix = np.cumsum(1 - labels) / rejection_count
        
        eer = 0.5 * (false_acceptance_ratio_suffix[np.argmin(np.abs(false_acceptance_ratio_suffix - false_rejection_ratio_prefix))] + false_rejection_ratio_prefix[np.argmin(np.abs(false_acceptance_ratio_suffix - false_rejection_ratio_prefix))])
        return float(eer)
        
class tDCFMetric(BaseMetric):
    def __init__(self, asv_score_file_dev, asv_score_file_eval, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asv_score_file_dev = asv_score_file_dev
        self.asv_score_file_eval = asv_score_file_eval

    def _compute_det_curve(self, target_scores, nontarget_scores):
        n_scores = target_scores.size + nontarget_scores.size
        all_scores = np.concatenate((target_scores, nontarget_scores))
        labels = np.concatenate(
            (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

        indices = np.argsort(all_scores, kind='mergesort')
        labels = labels[indices]
        tar_trial_sums = np.cumsum(labels)
        nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
        frr = np.concatenate(
            (np.atleast_1d(0), tar_trial_sums / target_scores.size))
        far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                            nontarget_scores.size))
        thresholds = np.concatenate(
            (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

        return frr, far, thresholds
    
    def _compute_eer(self, target_scores, nontarget_scores):
        frr, far, thresholds = self._compute_det_curve(target_scores, nontarget_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        return eer, thresholds[min_index]
    
    def _obtain_asv_error_rates(self, tar_asv, non_asv, spoof_asv, asv_threshold):
        Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
        Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

        if spoof_asv.size == 0:
            Pmiss_spoof_asv = None
        else:
            Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

        return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv
    
    def _compute_tDCF(self, bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv,
                 Pmiss_spoof_asv, cost_model):

        if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
                cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
            print('WARNING: Usually the cost values should be positive!')

        if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
                np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
            raise ValueError('Your prior probabilities should be positive and sum up to one.')

        if Pmiss_spoof_asv is None:
            raise Exception('you should provide miss rate of spoof tests against your ASV system.')

        combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
        if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
            raise ValueError('Your scores contain nan or inf.')

        n_uniq = np.unique(combined_scores).size
        if n_uniq < 3:
            raise ValueError('You should provide soft CM scores - not binary decisions')

        Pmiss_cm, Pfa_cm, CM_thresholds = self._compute_det_curve(
            bonafide_score_cm, spoof_score_cm)

        C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
            cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
        C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

        if C1 < 0 or C2 < 0:
            raise ValueError(
                'You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?'
            )

        tDCF = C1 * Pmiss_cm + C2 * Pfa_cm
        tDCF_norm = tDCF / np.minimum(C1, C2)
        return tDCF_norm, CM_thresholds

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, filenames, part, *args, **kwargs):
        
        DEFAULT_LA_PATH = ROOT_PATH / "src" / "data" / "LA"

        if not DEFAULT_LA_PATH.exists():
            raise LookupError("There's no default LA and LA_PATH is None")

        if part == 'val':
            asv_score_file = DEFAULT_LA_PATH / self.asv_score_file_dev
        elif part == 'test':
            asv_score_file = DEFAULT_LA_PATH / self.asv_score_file_eval

        Pspoof = 0.05
        cost_model = {
            'Pspoof': Pspoof,  # Prior probability of a spoofing attack
            'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
            'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
            'Cmiss': 1,  # Cost of ASV system falsely rejecting target speaker
            'Cfa': 10,  # Cost of ASV system falsely accepting nontarget speaker
            'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
            'Cfa_asv':
            10,  # Cost of ASV system falsely accepting nontarget speaker
            'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
            'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
        }

        asv_data = np.genfromtxt(asv_score_file, dtype=str)
        asv_keys = asv_data[:, 1]
        asv_scores = asv_data[:, 2].astype(np.float32)
        tar_asv = asv_scores[asv_keys == 'target']
        non_asv = asv_scores[asv_keys == 'nontarget']
        spoof_asv = asv_scores[asv_keys == 'spoof']
        eer_asv, asv_threshold = self._compute_eer(tar_asv, non_asv)
        cm_keys = labels
        cm_scores = torch.nn.functional.sigmoid(logits[:, 1] - logits[:, 0]).cpu().detach().numpy()
        bona_cm = cm_scores[cm_keys == 1]
        spoof_cm = cm_scores[cm_keys == 0]

        [Pfa_asv, Pmiss_asv,
        Pmiss_spoof_asv] = self._obtain_asv_error_rates(tar_asv, non_asv, spoof_asv,
                                                asv_threshold)

        tDCF_curve, CM_thresholds = self._compute_tDCF(bona_cm,
                                                spoof_cm,
                                                Pfa_asv,
                                                Pmiss_asv,
                                                Pmiss_spoof_asv,
                                                cost_model)

        min_tDCF_index = np.argmin(tDCF_curve)
        min_tDCF = tDCF_curve[min_tDCF_index]

        return float(min_tDCF)

