import os
import gzip
import torch
import urllib
import numpy as np
from PIL import Image
import torch.utils.data


class MovingMNIST(torch.utils.data.Dataset):
    """
    MovingMNIST dataset class.
    """

    urls = ["https://github.com/vineeths96/Video-Frame-Prediction/raw/master/data/mnist_test_seq.npy.gz"]

    raw_folder = "raw"
    processed_folder = "processed"
    training_file = "moving_mnist_train.pt"
    test_file = "moving_mnist_test.pt"

    def __init__(self, root, train=True, test_samples=1000, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.test_samples = test_samples
        self.train = train

        if download:
            self.download()

        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self.train:
            self.train_data = torch.load(os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data = torch.load(os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        if self.train:
            seq, target = self.train_data[index, :10], self.train_data[index, 10:]
        else:
            seq, target = self.test_data[index, :10], self.test_data[index, 10:]

        seq = [Image.fromarray(seq.numpy()[i, :, :], mode="L") for i in range(10)]
        target = [Image.fromarray(target.numpy()[i, :, :], mode="L") for i in range(10)]

        if self.transform is not None:
            seq = torch.stack([self.transform(seq[i]) for i in range(10)])

        if self.target_transform is not None:
            target = torch.stack([self.target_transform(target[i]) for i in range(10)])

        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and os.path.exists(
            os.path.join(self.root, self.processed_folder, self.test_file)
        )

    def download(self):
        if self.check_exists():
            return

        os.makedirs(os.path.join(self.root, self.raw_folder), exist_ok=True)
        os.makedirs(os.path.join(self.root, self.processed_folder), exist_ok=True)

        for url in self.urls:
            print("Downloading " + url)
            data = urllib.request.urlopen(url).read()
            filename = url.rpartition("/")[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)

            with open(file_path, "wb") as f:
                f.write(data)

            with open(file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        print("Processing...")

        train_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, "mnist_test_seq.npy")).swapaxes(0, 1)[
                : -self.test_samples
            ]
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), "wb") as f:
            torch.save(train_set, f)

        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, "mnist_test_seq.npy")).swapaxes(0, 1)[
                -self.test_samples :
            ]
        )

        with open(os.path.join(self.root, self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)

        print("Done!")

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of data points: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Train/Test: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp)))
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp)))

        return fmt_str
