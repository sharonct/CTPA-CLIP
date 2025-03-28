import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]

    resized_array = (
        F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False)
        .cpu()
        .numpy()
    )
    return resized_array


class CTReportDataset(Dataset):
    def __init__(
        self,
        data_folder,
        csv_file,
        split = 'train',
        min_slices=20,
        resize_dim=500,
        force_num_frames=True,
    ):
        self.split = split
        self.data_folder = data_folder  # Base directory: train_PE
        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths = []
        self.samples = self.prepare_samples()
        percent = 80
        num_files = int((len(self.samples) * percent) / 100)
        self.samples = self.samples[:num_files]
        print(f"Loaded {len(self.samples)} samples.")

        self.transform = transforms.Compose(
            [transforms.Resize((resize_dim, resize_dim)), transforms.ToTensor()]
        )
        self.npz_to_tensor = partial(self.npz_img_to_tensor, transform=self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row["impression_id"]] = row["impressions"]
        return accession_to_text

    def prepare_samples(self):
        samples = []

        for study_folder in tqdm.tqdm(
            glob.glob(os.path.join(self.data_folder, "*"))
        ):  # Iterate over parent folders
            study_id = os.path.basename(study_folder)

            for subfolder in glob.glob(
                os.path.join(study_folder, "*")
            ):  # Iterate over subdirectories
                if self.split == 'train':
                    filename = os.path.basename(subfolder).replace("train_", "")
                    npz_file = os.path.join(subfolder, filename + ".npz")
                else:
                    filename = os.path.basename(subfolder).replace("test_", "")
                    npz_file = os.path.join(subfolder, filename + ".npz")

                if not os.path.exists(npz_file):
                    continue

                accession_number = os.path.basename(npz_file).replace(".npz", "")
                if accession_number not in self.accession_to_text:
                    continue

                impression_text = self.accession_to_text[accession_number]
                input_text_concat = (
                    " ".join(impression_text) if impression_text != "Not given." else ""
                )

                samples.append((npz_file, input_text_concat))
                self.paths.append(npz_file)

        return samples

    def __len__(self):
        return len(self.samples)

    def npz_img_to_tensor(self, path, transform):
        data = np.load(path)
        ct_scan = data["arr_0"]
        if self.split == 'train':
            df = pd.read_csv("/teamspace/studios/this_studio/data/train_metadata.csv")
        else:
            df = pd.read_csv("/teamspace/studios/this_studio/data/test_metadata.csv")
            
        file_name = os.path.basename(path).replace("npz", "nii")

        row = df[file_name == df["VolumeName"]]

        if row.empty:
            raise ValueError(f"Metadata not found for {file_name}")

        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        img_data = slope * ct_scan + intercept
        img_data = np.transpose(img_data, (2, 0, 1))

        tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
        img_data = resize_array(
            tensor,
            (z_spacing, xy_spacing, xy_spacing),
            (target_z_spacing, target_x_spacing, target_y_spacing),
        )
        img_data = img_data[0][0]
        img_data = np.transpose(img_data, (1, 2, 0))

        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (img_data / 1000).astype(np.float32)

        tensor = torch.tensor(img_data)
        target_shape = (480, 480, 240)

        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        h_start, h_end = max((h - dh) // 2, 0), min((h - dh) // 2 + dh, h)
        w_start, w_end = max((w - dw) // 2, 0), min((w - dw) // 2 + dw, w)
        d_start, d_end = max((d - dd) // 2, 0), min((d - dd) // 2 + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before, pad_h_after = (
            (dh - tensor.size(0)) // 2,
            dh - tensor.size(0) - (dh - tensor.size(0)) // 2,
        )
        pad_w_before, pad_w_after = (
            (dw - tensor.size(1)) // 2,
            dw - tensor.size(1) - (dw - tensor.size(1)) // 2,
        )
        pad_d_before, pad_d_after = (
            (dd - tensor.size(2)) // 2,
            dd - tensor.size(2) - (dd - tensor.size(2)) // 2,
        )

        tensor = torch.nn.functional.pad(
            tensor,
            (
                pad_d_before,
                pad_d_after,
                pad_w_before,
                pad_w_after,
                pad_h_before,
                pad_h_after,
            ),
            value=-1,
        )
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        return tensor

    def __getitem__(self, index):
        npz_file, input_text = self.samples[index]
        video_tensor = self.npz_to_tensor(npz_file)

        input_text = (
            input_text.replace('"', "")
            .replace("'", "")
            .replace("(", "")
            .replace(")", "")
        )

        return video_tensor, input_text
