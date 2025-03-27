# Check if we are running this notebook on kaggle, google colab or local compute resources.
import os

import warnings
warnings.filterwarnings("ignore")

from glob import glob
from IPython.display import FileLink

import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components

import torch

from torch_em.util.debug import check_loader
from torch_em.data import MinInstanceSampler
from torch_em.util.util import get_random_colors

import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

root_dir = ""

image_dir = '/group/jug/Enrico/TISSUE_roi_projection'
image_paths = sorted(glob(os.path.join(image_dir, "*/*_GFP_max.tif")))

segmentation_dir = '/group/jug/Enrico/TISSUE_roi_projection'
segmentation_paths = sorted(glob(os.path.join(image_dir, "*/*_CELL_comb.tif")))

# let's visualize how the samples look

for image_path, segmentation_path in zip(image_paths, segmentation_paths):
    image_name = image_path.split('/')[-2]
    image = imageio.imread(image_path)
    print(image.shape)
    segmentation = imageio.imread(segmentation_path)
    print(segmentation.shape)

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))

    ax[0].imshow(image, cmap="gray", vmin=np.amin(image), vmax=np.amax(image)/4)
    ax[0].set_title(image_name)
    ax[0].axis("off")

    segmentation = connected_components(segmentation)
    ax[1].imshow(segmentation, cmap=get_random_colors(segmentation), interpolation="nearest")
    ax[1].set_title("Ground Truth Instances")
    ax[1].axis("off")

    # save the figure
    plt.tight_layout()
    plt.savefig(os.path.join('data_summary', image_name + '.png'))

    plt.show()
    plt.close()

    break  # comment this out in case you want to visualize all the images


# Load images from multiple files in folder via pattern (here: all tif files)
raw_key, label_key = "*_GFP_max.tif", "*_CELL_comb.tif"

# Alternative: if you have tif stacks you can just set 'raw_key' and 'label_key' to None
# raw_key, label_key= None, None

'''# The 'roi' argument can be used to subselect parts of the data.
# Here, we use it to select the first 70 images (frames) for the train split and the other frames for the val split.
train_roi = np.s_[:70, :, :]
val_roi = np.s_[70:, :, :]'''
# already splitted into two folders
training_dir = os.path.join(image_dir, "training")
validation_dir = os.path.join(image_dir, "validation")

# normalization function
from torch_em.transform.raw import normalize_percentile
def normalization_8bit(image):
    image = normalize_percentile(image)  # Use 1st and 99th percentile values for min-max normalization.
    image = np.clip(image, 0, 1)  # Clip the values to be in range [0, 1].

    image *= 255
    image = image.astype(np.uint8)
    return image

# Here, we use `micro_sam.training.default_sam_loader` for creating a suitable data loader from
# the example hela data. You can either adapt this for your own data or write a suitable torch dataloader yourself.
# Here's a quickstart notebook to create your own dataloaders: https://github.com/constantinpape/torch-em/blob/main/notebooks/tutorial_create_dataloaders.ipynb

batch_size = 1  # the training batch size
patch_shape = (1, 512, 512)  # the size of patches for training

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
    raw_transform=normalization_8bit,
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
    raw_transform=normalization_8bit,
    sampler=sampler,
)

# All hyperparameters for training.
n_objects_per_batch = 5  # the number of objects per batch that will be sampled
device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training
n_epochs = 5  # how long we train (in epochs)

# The model_type determines which base model is used to initialize the weights that are finetuned.
# We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
model_type = "vit_t_lm"

# The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
checkpoint_name = "sam_hela"

# run training
sam_training.train_sam(
    name=checkpoint_name,
    save_root=os.path.join(root_dir, "models"),
    model_type=model_type,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=n_epochs,
    n_objects_per_batch=n_objects_per_batch,
    with_segmentation_decoder=train_instance_segmentation,
    device=device,
)

# Let's spot our best checkpoint and download it to get started with the annotation tool
best_checkpoint = os.path.join("models", "checkpoints", checkpoint_name, "best.pt")

# Download link is automatically generated for the best model.
print("Click here \u2193")
FileLink(best_checkpoint)