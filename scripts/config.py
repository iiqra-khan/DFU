import torch

class Config:
    # Paths
    FUSEG_PATH = '../data/fuseg'
    DPM_PATH = '../data/dpm_v3'

    # Model
    BACKBONE = 'efficientnet-b4'  # For Stage 2 only; Stage 1 uses SegFormer-B2
    TIMM_BACKBONE = 'efficientnet_b4'
    ENCODER_WEIGHTS = 'imagenet'
    
    # Stage 1: SegFormer-B2
    SEGFORMER_MODEL = 'nvidia/segformer-b2-finetuned-ade-512-512'
    SEGFORMER_NUM_LABELS = 1

    # Training
    BATCH_SIZE = 16
    EPOCHS_STAGE1 = 20
    EPOCHS_STAGE2 = 50
    LR_STAGE1 = 3e-4
    LR_STAGE2 = 1e-5
    WEIGHT_DECAY = 1e-4
    
    # Optimizer & Scheduler
    OPTIMIZER_STAGE1 = 'adamw'
    USE_SCHEDULER_STAGE1 = True
    SCHEDULER_STAGE1 = 'cosine'
    
    # Loss for Stage 1
    LOSS_STAGE1 = 'dice_bce'
    LOSS_WEIGHTS = {'bce': 0.3, 'dice': 0.7}

    # Early stopping
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_MIN_DELTA = 1e-4
    EARLY_STOPPING_PATIENCE_STAGE1 = 5
    EARLY_STOPPING_PATIENCE_STAGE2 = 7

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