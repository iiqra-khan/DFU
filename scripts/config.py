import torch

class Config:
    # Paths
    FUSEG_PATH = '../data/fuseg'
    DPM_PATH = '../data/dpm_v3'

    # Model
    BACKBONE = 'resnet50'  # or 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'

    # Training
    BATCH_SIZE = 16
    EPOCHS_STAGE1 = 30
    EPOCHS_STAGE2 = 50
    LR_STAGE1 = 1e-4
    LR_STAGE2 = 1e-5

    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4

    # Output
    OUTPUT_DIR = './outputs'

    # Experiment tracking (optional)
    USE_WANDB = False
    WANDB_PROJECT = 'dfu-pipeline'
    WANDB_ENTITY = None  # set to your wandb username/team for shared projects
    WANDB_RUN_NAME = None