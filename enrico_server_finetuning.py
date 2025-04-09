import warnings
warnings.filterwarnings("ignore")

import os
import argparse
from glob import glob

import torch
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training



def finetuning(root_dir, raw_dataset, segmentation_gt, model):

    #image_dir = '/group/jug/Enrico/TISSUE_roi_projection'
    image_dir = root_dir
    image_paths = sorted(glob(os.path.join(image_dir, '*/*_' + raw_dataset + '.tif')))

    #segmentation_dir = '/group/jug/Enrico/TISSUE_roi_projection'
    segmentation_dir = root_dir
    segmentation_paths = sorted(glob(os.path.join(image_dir, "*/*_" + segmentation_gt + ".tif")))


    # Load images from multiple files in folder via pattern (here: all tif files)
    raw_key, label_key = '*_' + raw_dataset + '.tif', '*_' + segmentation_gt + '.tif'



    # already splitted into two folders
    training_dir = os.path.join(image_dir, "training")
    validation_dir = os.path.join(image_dir, "validation")

    # Here, we use `micro_sam.training.default_sam_loader` for creating a suitable data loader from
    # the example hela data. You can either adapt this for your own data or write a suitable torch dataloader yourself.
    # Here's a quickstart notebook to create your own dataloaders: https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb

    batch_size = 1  # the training batch size
    patch_shape = (1, 1024, 1024)  # the size of patches for training

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

    train_loader = sam_training.default_sam_loader(
        raw_paths=training_dir,
        raw_key=raw_key,
        label_paths=training_dir,
        label_key=label_key,
        with_segmentation_decoder=train_instance_segmentation,
        patch_shape=patch_shape,
        batch_size=batch_size,
        is_seg_dataset=True,
        shuffle=True,
        raw_transform=sam_training.identity,
        sampler=sampler,
    )

    val_loader = sam_training.default_sam_loader(
        raw_paths=validation_dir,
        raw_key=raw_key,
        label_paths=validation_dir,
        label_key=label_key,
        with_segmentation_decoder=train_instance_segmentation,
        patch_shape=patch_shape,
        batch_size=batch_size,
        is_seg_dataset=True,
        shuffle=True,
        raw_transform=sam_training.identity,
        sampler=sampler,
    )

    # All hyperparameters for training.
    n_objects_per_batch = 5  # the number of objects per batch that will be sampled
    device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training
    n_epochs = 5  # how long we train (in epochs)

    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
    model_type = model

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = model + '---' + raw_dataset + '---' + segmentation_gt

    if not os.path.exists(os.path.join(root_dir, 'microsam_models')):
        os.makedirs(os.path.join(root_dir, 'microsam_models'))

    sam_training.train_sam(
        name=checkpoint_name,
        save_root=os.path.join(root_dir, 'microsam_models'),
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=train_instance_segmentation,
        device=device,
    )




raw_ds = ('GFP_max', 'GFP_max_clahe')
seg_ds = ('CELL_max', 'CELL_comb')

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
