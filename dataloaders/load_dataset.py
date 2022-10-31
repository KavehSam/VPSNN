import os
import os.path

import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset


class ECGDataset(Dataset):
    """The source of the data is the benchmark MIT-BIH Arrhythmia Database from PhysioNet.
       Only the modified limb lead II channel was used from each recording,
       which was preprocessed according to [1] as follows:
         (I)  Filtering basic noise types:
               (a) Baseline removal by wavelets,
               (b) High-frequency noise removal by 35Hz low-pass filters.
         (II)  QRS segmentation:
               (a) R peak localization based on the reference annotations of the dataset,
               (b) Windowing the QRS complexes considering 100 number of samples centered around the R peaks.
         (III) Train and test sets:
               (a) QRS complexes that correspond to Normal (N) and Ventricular Ectopic beats (V) were selected.
                   Then, the normal class were subsampled to balance the train and test data, i.e.,
                   in order to have as many normal as ectopic QRS complexes in each class.
               (b) The subsampled QRS complexes were stored in mat files. Each mat file contains
                   the 'samples' of the QRS complexes, and the class 'labels' in one-hot encoding.
               (c) The QRS complexes were split into sets DS1 and DS2 according to [2],
                   for training and inference, respectively. The main feature of this separation is that
                   the recordings of DS1 (train) and DS2 (test) come from different patients,
                   which means that there is no data leakage in either cases.

       References:
         [1]  P. Kovács, G. Bognár, C. Huber, M. Huemer, VPNET: Variable Projection Networks,
              International Journal of Neural Systems (IJNS), 2021, vol. 32, no. 1, pp. 2150054:1-19.

         [2]  P. de Chazal, M. O’Dwyer and R. B. Reilly, Automatic classification of heartbeats using
              ECG morphology and heartbeat interval features, IEEE Trans. Biomed. Eng. 51(7) (2004) 1196–1206.
    """

    def __init__(self, dataset_path, transform=None):
        """Args:
            dataset_path (string): Path to the mat file that contains both the QRS complexes and the target labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataset = loadmat(dataset_path)
        self.dataset_path = dataset_path
        self.transform = transform

    def __len__(self):
        return self.dataset['samples'].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        qrs = self.dataset['samples'][idx, :].reshape(1, -1)
        target_label = self.dataset['labels'][idx].reshape(1, -1)  # [0]
        sample = [torch.from_numpy(qrs).float(), torch.from_numpy(target_label).float()]

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_ecg_real(batch_size, data_path):
    """Args:
        batch_size (int): the dataset will be split into mini-batches of this size.
        data_path (string): Path to the mat file that contains both the QRS complexes and the target labels.
    """
    transform = None

    train_mat_path = os.path.join(data_path, 'ecg_train.mat')
    trainset = ECGDataset(train_mat_path, transform=transform)

    test_mat_path = os.path.join(data_path, 'ecg_test.mat')
    testset = ECGDataset(test_mat_path, transform=transform)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size,
                            drop_last=False, shuffle=False, num_workers=0)

    return trainloader, testloader
