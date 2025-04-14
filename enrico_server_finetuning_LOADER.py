import warnings
warnings.filterwarnings("ignore")

import os
import argparse
from glob import glob

import torch
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training

import albumentations as A

from torch_em.util.debug import check_loader
from torch.utils.data import Dataset
from scipy.ndimage import binary_erosion
import numpy as np
from skimage.measure import label
from skimage.io import imread
import torch_em

from torch.utils.data import DataLoader


# create a torch dataset
# create a torch dataset
class CellDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        crop_size=None,
        padding_size=8,
        with_segmentation_decoder=True,
        augmentation=False):

      self.images = list(sorted(glob(image_dir)))
      self.masks = list(sorted(glob(mask_dir)))

      self.crop_size = crop_size
      self.padding_size = padding_size
      self.with_segmentation_decoder = with_segmentation_decoder
      self.augmentation = augmentation

      if len(self.images) == 0 or len(self.images) != len(self.masks):
          raise Exception('something wrong with the directory')


    def __len__(self):
      return len(self.images)

    # function to erode label boundaries
    def erode(self, labels, iterations, border_value):

      foreground = np.zeros_like(labels, dtype=bool)

      # loop through unique labels
      for label in np.unique(labels):

          # skip background
          if label == 0:
              continue

          # mask to label
          label_mask = labels == label

          # erode labels
          eroded_mask = binary_erosion(
                  label_mask,
                  iterations=iterations,
                  border_value=border_value)

          # get foreground
          foreground = np.logical_or(eroded_mask, foreground)

      # and background...
      background = np.logical_not(foreground)

      # set eroded pixels to zero
      labels[background] = 0

      return labels

    # takes care of padding
    def get_padding(self, crop_size, padding_size):
    
        # quotient
        q = int(crop_size / padding_size)
    
        if crop_size % padding_size != 0:
            padding = (padding_size * (q + 1))
        else:
            padding = crop_size
    
        return padding
    
    # sample augmentations (see https://albumentations.ai/docs/examples/example_kaggle_salt)
    def augment_data(self, raw, mask, padding):
        
        transform = A.Compose([
              A.RandomCrop(
                  width=self.crop_size,
                  height=self.crop_size),
              A.PadIfNeeded(
                  min_height=padding,
                  min_width=padding,
                  p=1,
                  border_mode=0),
              A.HorizontalFlip(p=0.3),
              A.VerticalFlip(p=0.3),
              A.RandomRotate90(p=0.3),
              A.Transpose(p=0.3),
              A.RandomBrightnessContrast(p=0.3)
            ])

        transformed 
        transformed = transform(image=raw, mask=mask)

        raw, mask = transformed['image'], transformed['mask']

        # I guess that the training and loader deals with 3d pictures, so it wants another axis
        if len(raw.shape) == 2:
            raw = np.expand_dims(raw, axis=0)
            #mask = np.expand_dims(mask, axis=0)
        
        return raw, mask

    # normalize raw data between 0 and 1
    def normalize(self, data):
      return ((data - np.min(data)) / (np.max(data) - np.min(data))*255).astype(np.float32)

    def __getitem__(self, idx):

      raw = self.images[idx]
      labels = self.masks[idx]

      raw = imread(raw)
      raw = self.normalize(raw)
      
      # slice first channel, relabel connected components
      labels = label(imread(labels)).astype(np.uint16)

      padding = self.get_padding(self.crop_size, self.padding_size)
      if self.augmentation:
        raw, labels = self.augment_data(raw, labels, padding)

      min_size = 25
      if self.with_segmentation_decoder:
        label_transform = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=True,
            min_size=min_size,
        )
      else:
        label_transform = torch_em.transform.label.MinSizeLabelTransform(min_size=min_size)
      labels = label_transform(labels).astype(np.float32)

      return raw, labels


def finetuning(root_dir, raw_dataset, segmentation_gt, model):

    #root_dir = '/group/jug/Enrico/TISSUE_roi_projection'


    # Load images from multiple files in folder via pattern (here: all tif files)
    raw_key = '*_' + raw_dataset + '.tif'
    label_key = '*_' + segmentation_gt + '.tif'



    # already splitted into two folders
    training_dir = os.path.join(root_dir, "training")
    validation_dir = os.path.join(root_dir, "validation")

    # Here, we use `micro_sam.training.default_sam_loader` for creating a suitable data loader from
    # the example hela data. You can either adapt this for your own data or write a suitable torch dataloader yourself.
    # Here's a quickstart notebook to create your own dataloaders: https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb

    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    # NOTE 1: It's important to have densely annotated-labels while training the additional convolutional decoder.
    # NOTE 2: In case you do not have labeled images, we recommend using `micro-sam` annotator tools to annotate as many objects as possible per image for best performance.
    train_instance_segmentation = True

    # NOTE: The dataloader internally takes care of adding label transforms: i.e. used to convert the ground-truth
    # labels to the desired instances for finetuning Segment Anythhing, or, to learn the foreground and distances
    # to the object centers and object boundaries for automatic segmentation.

    # There are cases where our inputs are large and the labeled objects are not evenly distributed across the image.
    # For this we use samplers, which ensure that valid inputs are chosen subjected to the paired labels.
    # The sampler chosen below makes sure that the chosen inputs have atleast one foreground instance, and filters out small objects.
    sampler = MinInstanceSampler(min_size=25)  # NOTE: The choice of 'min_size' value is paired with the same value in 'min_size' filter in 'label_transform'.

    n_objects_per_batch = 1  # the number of objects per batch that will be sampled
    crop_size = 800  # the size of patches

    # train dataset and loader, with augmentation
    train_dataset = CellDataset(
        image_dir = os.path.join(training_dir, raw_key),
        mask_dir = os.path.join(training_dir, label_key),
        crop_size=crop_size, with_segmentation_decoder=True, augmentation=True)
    train_loader = DataLoader(train_dataset, batch_size=n_objects_per_batch, shuffle=True)
    train_loader.shuffle = True

    # validation dataset and loadet, no augmentation
    valid_dataset = CellDataset(
        image_dir = os.path.join(validation_dir, raw_key),
        mask_dir = os.path.join(validation_dir, label_key),
        crop_size=crop_size, with_segmentation_decoder=True, augmentation=True)
    valid_loader = DataLoader(valid_dataset, batch_size=n_objects_per_batch, shuffle=True)
    valid_loader.shuffle = True


    # All hyperparameters for training.

    device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training
    n_epochs = 5  # how long we train (in epochs)

    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
    model_type = model

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = model + '---' + raw_dataset + '---' + segmentation_gt

    model_folder = 'microsam_models_TEST'

    if not os.path.exists(os.path.join(root_dir, model_folder)):
        os.makedirs(os.path.join(root_dir, model_folder))

    sam_training.train_sam(
        name=checkpoint_name,
        save_root=os.path.join(root_dir, model_folder),
        model_type=model_type,
        train_loader=train_loader,
        val_loader=valid_loader,
        n_epochs=n_epochs,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=train_instance_segmentation,
        device=device,
    )



raw_ds = ('GFP_max', 'GFP_max_clahe', 'GFP_sum')
seg_ds = ('CELL_max', 'CELL_comb', 'CELL_unique', 'CELL_manual')

models = ('vit_t_lm', 'vit_b_lm', 'vit_l_lm')

parser = argparse.ArgumentParser(
                    prog='microsam_2d_finetuning',
                    description='finetune a microsam model with specific 2d dataset and annotation')

#parser.add_argument('filename')
parser.add_argument('-r', '--raw')
parser.add_argument('-s', '--segmentation')
parser.add_argument('-m', '--model')

args = parser.parse_args()

r, s, m = args.raw, args.segmentation, args.model

if (r not in raw_ds):
    raise Exception('raw not found')
if (s not in seg_ds):
    raise Exception('segmentation not found')
if (m not in models):
    raise Exception('model not present')

print()
print(f'finetuning {m} model with {r} images, gt is {s}')
print()

dataset_dir = '/group/jug/Enrico/TISSUE_roi_projection'

finetuning(dataset_dir, r, s, m)

'''
/localscratch/conda/envs/napari-env/bin/python /home/enrico.negri/github/micro-sam-finetuning/enrico_server_finetuning_LOADER.py -r 'GFP_max' -s 'CELL_max' -m 'vit_t_lm'
'''