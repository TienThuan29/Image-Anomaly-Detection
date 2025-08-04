import torch
from torchmetrics import ROC, AUROC, F1Score
from skimage import measure
from sklearn.metrics import roc_auc_score, roc_curve


class Metric:
    def __init__(self, labels_list, predictions, anomaly_map_list, gt_list) -> None:
        self.labels_list = labels_list
        self.predictions = predictions
        self.anomaly_map_list = anomaly_map_list
        self.gt_list = gt_list
        # self.threshold = 0.5

    def image_auroc(self):
        auroc_image = roc_auc_score(self.labels_list, self.predictions)
        return auroc_image

    def pixel_auroc(self):
        resutls_embeddings = self.anomaly_map_list[0]
        for feature in self.anomaly_map_list[1:]:
            resutls_embeddings = torch.cat((resutls_embeddings, feature), 0)
        resutls_embeddings = ((resutls_embeddings - resutls_embeddings.min()) / (
                    resutls_embeddings.max() - resutls_embeddings.min()))

        gt_embeddings = self.gt_list[0]
        for feature in self.gt_list[1:]:
            gt_embeddings = torch.cat((gt_embeddings, feature), 0)

        resutls_embeddings = resutls_embeddings.clone().detach().requires_grad_(False)
        gt_embeddings = gt_embeddings.clone().detach().requires_grad_(False)

        auroc_p = AUROC(task="binary")

        gt_embeddings = torch.flatten(gt_embeddings).type(torch.bool).cpu().detach()
        resutls_embeddings = torch.flatten(resutls_embeddings).cpu().detach()
        auroc_pixel = auroc_p(resutls_embeddings, gt_embeddings)
        return auroc_pixel


