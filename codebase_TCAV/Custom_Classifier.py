
import random
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from captum.concept._utils.classifier import Classifier
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class EfficientClassifier(Classifier):

    def __init__(self, device='cpu', epochs=30, batch_size=200) -> None:
        self.lm = SGDClassifier(alpha=0.01, max_iter=5000, tol=1e-3)
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

    def train_and_eval(
        self, dataloader: DataLoader, test_split_ratio: float = 0.2, **kwargs: Any
    ) -> Union[Dict, None]:
        self.__init__()
        len_data = len(dataloader)
        data_idxs = list(range(len_data))

        train_idx, test_idx = train_test_split(
            data_idxs, test_size=test_split_ratio, shuffle=True)

        train_set = Subset(dataloader.dataset, train_idx)
        test_set = Subset(dataloader.dataset, test_idx)

        train_dataloader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=True)

        # train cycle
        for i in range(self.epochs):
            for idx, data in enumerate(train_dataloader):
                batch = data[0]
                labels = data[1].cpu()
                batch = torch.squeeze(batch, dim=1).cpu()

                if len(train_dataloader) == 1:
                    self.lm.fit(batch, labels)
                else:
                    self.lm.partial_fit(batch, labels, np.unique(labels))

        # test_cycle
        batch_acc = []
        for idx, data in enumerate(test_dataloader):
            batch = data[0]
            labels = data[1]

            batch = torch.squeeze(batch, dim=1).cpu()
            labels = torch.squeeze(labels).cpu()

            pred_labels = self.lm.predict(batch)

            batch_acc.append(
                (labels.tolist() == pred_labels).sum())

        accs = round(sum(batch_acc)/len(test_idx), 2)
        self.save_classes = np.unique(labels)
        return {"accs": accs, "f1_score": 0}

    def weights(self) -> Tensor:
        r"""
        This function returns a C x F tensor weights, where
        C is the number of classes and F is the number of features.
        In case of binary classification, C = 2 otherwise it is > 2.

        Returns:
            weights (Tensor): A torch Tensor with the weights resulting from
                the model training.
        """

        weights = torch.tensor(self.lm.coef_)

        if weights.shape[0] == 1:
            # if there are two concepts, there is only one label. We split it in two.
            return torch.stack([-1 * weights[0], weights[0]])
        else:
            return weights

    def classes(self) -> List[int]:
        r"""
        This function returns the list of all classes that are used by the
        classifier to train the model in the `train_and_eval` method.
        The order of returned classes has to match the same order used in
        the weights matrix returned by the `weights` method.

        Returns:
            classes (list): The list of classes used by the classifier to train
            the model in the `train_and_eval` method.
        """
        return self.save_classes.tolist()  # type: ignore


class StratifiedSVMClassifier(Classifier):

    def __init__(self) -> None:
        self.lm = SGDClassifier(alpha=0.1, max_iter=100, verbose=1)
        self.model_pipeline = make_pipeline(StandardScaler(), self.lm)

    def train_and_eval(
            self, dataloader: DataLoader, test_split_ratio: float = 0.2, **kwargs: Any) -> Union[Dict, None]:
        self.__init__()

        inputs = []
        labels = []
        for input, label in dataloader:
            inputs.append(input.squeeze().cpu().detach().numpy())
            labels.append(label.squeeze().cpu().detach().numpy())

        all_examples = np.array(inputs)
        all_labels = np.array(labels)
        self.save_classes = np.unique(all_labels)

        train_features, test_features, train_labels, test_labels = train_test_split(all_examples, all_labels,
                                                                                    test_size=test_split_ratio,
                                                                                    shuffle=True,
                                                                                    stratify=all_labels)

        print("Length of training set: {}".format(train_features.shape[0]), flush=True)
        self.model_pipeline.fit(train_features, train_labels)

        train_preds = self.model_pipeline.predict(train_features)
        print(confusion_matrix(train_labels, train_preds))

        print("Length of test set: {}".format(test_features.shape[0]))
        test_preds = self.model_pipeline.predict(test_features)
        accuracy = accuracy_score(test_labels, test_preds)
        f1 = f1_score(test_labels, test_preds, average="binary", pos_label=np.max(all_labels))
        print("Accuracy on the training set: {}".format(accuracy_score(train_labels, train_preds)))
        print("Accuracy on the test set: {}".format(accuracy))
        print(confusion_matrix(test_labels, test_preds))

        return {"accs": accuracy, "f1_score": f1}

    def weights(self) -> Tensor:
        r"""
        This function returns a C x F tensor weights, where
        C is the number of classes and F is the number of features.
        In case of binary classification, C = 2 otherwise it is > 2.

        Returns:
            weights (Tensor): A torch Tensor with the weights resulting from
                the model training.
        """

        weights = torch.tensor(self.lm.coef_)
        if weights.shape[0] == 1:
            # if there are two concepts, there is only one label. We split it in two.
            return torch.stack([-1 * weights[0], weights[0]])
        else:
            return weights

    def classes(self) -> List[int]:
        r"""
        This function returns the list of all classes that are used by the
        classifier to train the model in the `train_and_eval` method.
        The order of returned classes has to match the same order used in
        the weights matrix returned by the `weights` method.

        Returns:
            classes (list): The list of classes used by the classifier to train
            the model in the `train_and_eval` method.
        """
        return self.save_classes.tolist()  # type: ignore
