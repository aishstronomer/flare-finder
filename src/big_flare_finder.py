from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class BigFlareFinder:
    def __init__(self):

        # init pytorch model
        self.pytorch_model = None

        # set things to make training deterministic
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

        # set device
        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def fit(self, image_paths, image_labels, val_frac=0.5):

        # split the data into train and validation using time
        image_paths_train, image_paths_val, image_labels_train, image_labels_val = (
            BigFlareFinder.split_into_train_test_by_time(
                image_paths, image_labels, val_frac
            )
        )

        # for training, augment minority-class by making copies
        image_paths_train, image_labels_train = BigFlareFinder.augment_minority_class(
            image_paths_train, image_labels_train
        )

        # get dataloader for train and validation data
        train_loader = BigFlareFinder.preprocess(image_paths_train, image_labels_train)
        validation_loader = BigFlareFinder.preprocess(image_paths_val, image_labels_val)

        # fit pytorch model using dataloader

        # load resnet18 model
        # model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        model_resnet18 = torch.hub.load(
            "pytorch/vision", "resnet18", weights=ResNet18_Weights.DEFAULT
        )

        # Freeze all params except the BatchNorm layers, as here they are trained to the
        # mean and standard deviation of ImageNet and we may lose some signal
        for name, param in model_resnet18.named_parameters():
            if "bn" not in name:
                param.requires_grad = False

        # reduce number of output classes in model
        num_classes = 2
        model_resnet18.fc = nn.Sequential(
            nn.Linear(model_resnet18.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        model_resnet18.to(self.device)
        optimizer = optim.Adam(model_resnet18.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()
        epochs = 7  # 10
        target_class = 1

        for epoch in range(epochs):
            training_loss = 0.0
            valid_loss = 0.0
            model_resnet18.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                targets = targets.type(torch.LongTensor)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = model_resnet18(inputs)
                loss = loss_fn(output, targets)

                loss.backward()
                optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
            training_loss /= len(train_loader.dataset)

            model_resnet18.eval()
            # all_targets = []
            image_labels_val_pred = []

            for batch in validation_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                output = model_resnet18(inputs)
                predictions = torch.max(F.softmax(output, dim=1), dim=1)[1]
                image_labels_val_pred.extend(predictions.cpu().numpy())

            print(
                f"Train epoch: {epoch}"
                f", train_loss: {round(training_loss, 2)}"
                f"\nval_metrics: {BigFlareFinder.get_model_performance_metrics(image_labels_val, image_labels_val_pred)}"
            )

        # train on the val data (which was excluded from training earlier)
        image_paths_val, image_labels_val = BigFlareFinder.augment_minority_class(
            image_paths_val, image_labels_val
        )
        val_loader = BigFlareFinder.preprocess(image_paths_val, image_labels_val)
        for epoch in range(epochs):
            for batch in val_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                targets = targets.type(torch.LongTensor)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                output = model_resnet18(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
            training_loss /= len(train_loader.dataset)
            model_resnet18.eval()
            print(
                f"Train-on-val epoch: {epoch}"
                f", train_loss: {round(training_loss, 2)}"
            )

        # save trained model to self
        self.pytorch_model = model_resnet18

    @staticmethod
    def augment_minority_class(image_paths, image_labels):
        # TODO: remove funtion after implementing augmentation in preprocess

        # init resut
        image_paths_aug = []
        image_labels_aug = []

        # augment data
        image_labels_counts = pd.Series(image_labels).value_counts().sort_values()
        minority_class, majority_class = tuple(image_labels_counts.index)
        class_count_diff = (
            image_labels_counts[majority_class] - image_labels_counts[minority_class]
        )
        image_paths_new = (
            pd.Series(image_paths[image_labels == minority_class])
            .sample(class_count_diff, replace=True, random_state=42)
            .to_list()
        )
        image_labels_new = [minority_class] * class_count_diff
        image_paths_aug = image_paths + image_paths_new
        image_labels_aug = image_labels + image_labels_new

        # shuffle augmented data
        image_paths_aug, image_labels_aug = zip(
            *np.random.default_rng(seed=42).permutation(
                list(zip(image_paths_aug, image_labels_aug))
            )
        )
        image_paths_aug = [str(path) for path in image_paths_aug]
        image_labels_aug = [float(label) for label in image_labels_aug]

        return image_paths_aug, image_labels_aug

    def predict(self, image_paths):

        # init result
        pred_labels = []

        # get dataloader
        dataloader = BigFlareFinder.preprocess(image_paths)

        # make predictions
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            output = self.pytorch_model(inputs)
            predictions = torch.max(F.softmax(output, dim=1), dim=1)[1]
            pred_labels.extend(predictions.cpu().numpy())

        return pred_labels

    def pred_proba(self, image_paths):

        # init result
        pred_probas = []

        # get dataloader
        dataloader = BigFlareFinder.preprocess(image_paths)

        # make predictions
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            output = self.pytorch_model(inputs)
            predictions = F.softmax(output, dim=1)[:, 1]

            # pdb.set_trace()

            pred_probas.extend(predictions.detach().cpu().numpy().tolist())

        return pred_probas

    @staticmethod
    def get_model_performance_metrics(y_true, y_pred):
        metrics_dict = {
            "accuracy": round(accuracy_score(y_true, y_pred), 2),
            "f1": round(f1_score(y_true, y_pred), 2),
            "precision_class_1": round(
                precision_score(y_true, y_pred, zero_division=0), 2
            ),
            "recall_class_1": round(recall_score(y_true, y_pred, zero_division=0), 2),
            "actual_distru": pd.Series(y_true).value_counts().sort_index().to_dict(),
            "pred_distru": pd.Series(y_pred).value_counts().sort_index().to_dict(),
        }
        return metrics_dict

    @staticmethod
    def preprocess(image_paths, image_labels=None, augment_minority_class=False):
        # TODO: add data augmentation option; see https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch

        # make dataset of image_paths and image_labels
        image_dimension = 224
        image_transforms = transforms.Compose(
            [
                transforms.Resize((image_dimension, image_dimension)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image_labels = image_labels or [0] * len(image_paths)
        dataset = CustomImageDataset(image_paths, image_labels, image_transforms)

        # make dataloader for dataset
        batch_size = 32
        num_workers = 2
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return dataloader

    @staticmethod
    def split_into_train_test_by_time(image_paths, image_labels, test_frac):
        # make images_df
        images_df = pd.DataFrame(
            {"image_path": image_paths, "image_label": image_labels}
        )

        # add a datetime column
        images_df["datetime"] = pd.to_datetime(
            images_df["image_path"].str.split("/").str[-1].str[None:-4],
            format = "mixed"
        )

        # sort images_df by datetime
        images_df = images_df.sort_values("datetime")

        # find min and max datetimes of images_df
        min_datetime = images_df["datetime"].min()
        max_datetime = images_df["datetime"].max()

        # find time span between min and max datetimes (in days)
        time_span_in_days = (max_datetime - min_datetime).days

        # get the two dfs by splitting the time span into the desired ratio
        num_train_days = time_span_in_days * (1 - test_frac)
        end_train_datetime = min_datetime + pd.to_timedelta(num_train_days, unit="days")
        train_df = images_df[images_df["datetime"] <= end_train_datetime]
        test_df = images_df[images_df["datetime"] > end_train_datetime]

        # get image_paths_train, image_paths_test, image_labels_train, image_labels_test
        image_paths_train = train_df["image_path"].to_list()
        image_paths_test = test_df["image_path"].to_list()
        image_labels_train = train_df["image_label"].to_list()
        image_labels_test = test_df["image_label"].to_list()

        return (
            image_paths_train,
            image_paths_test,
            image_labels_train,
            image_labels_test,
        )
