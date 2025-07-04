import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import CUBSATDataset
from utils import(
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_list_boxes,
    get_bboxes,
    save_checkpoint,
    load_checkpoint
)

from loss import YOLOv1Loss

seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 5e-6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {DEVICE}")
BATCH_SIZE = 32
WEIGHT_DECAY = 0


