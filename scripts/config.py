import torch

class Config:
    # Paths
    FUSEG_PATH = '../data/fuseg'
    DPM_PATH = '../data/dpm_v3'

    # Model
    # Stage 1: SegFormer-B2 for segmentation
    SEGFORMER_MODEL = 'nvidia/segformer-b2-finetuned-ade-512-512'
    SEGFORMER_NUM_LABELS = 1
    
    # Stage 2: SegFormer-B2 for 4-class Wagner classification
    SEGFORMER_NUM_CLASSES_STAGE2 = 4
    ENCODER_WEIGHTS = 'imagenet'  # Legacy reference
    
    # Class weights for DPM Wagner grades (approximate from dataset)
    # Adjust these based on actual class distribution in your DPM dataset
    CLASS_WEIGHTS_STAGE2 = torch.tensor([1.0, 1.2, 1.5, 1.8])

    # Training
    # Default batch size (reduce if OOMs occur). Adjust with GRADIENT_ACCUMULATION_STEPS
    BATCH_SIZE = 2
    EPOCHS_STAGE1 = 20
    EPOCHS_STAGE2 = 30
    LR_STAGE1 = 3e-4
    LR_STAGE2 = 1e-5
    WEIGHT_DECAY = 1e-4
    
    # Optimizer & Scheduler for Stage 1
    OPTIMIZER_STAGE1 = 'adamw'
    USE_SCHEDULER_STAGE1 = True
    SCHEDULER_STAGE1 = 'cosine'
    
    # Optimizer & Scheduler for Stage 2
    OPTIMIZER_STAGE2 = 'adamw'
    USE_SCHEDULER_STAGE2 = True
    SCHEDULER_STAGE2 = 'cosine'
    
    # Progressive layer unfreezing for Stage 2
    # Gradually unfreeze encoder blocks after this epoch
    UNFREEZE_ENCODER_EPOCH_STAGE2 = 5
    # Unfreeze one encoder block every N epochs
    UNFREEZE_INTERVAL_EPOCHS = 3
    
    # Loss for Stage 1
    LOSS_STAGE1 = 'dice_bce'
    LOSS_WEIGHTS = {'bce': 0.3, 'dice': 0.7}
    
    # Loss for Stage 2
    LOSS_STAGE2 = 'weighted_cross_entropy'
    USE_CLASS_WEIGHTS_STAGE2 = True

    # Early stopping
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_MIN_DELTA = 1e-4
    EARLY_STOPPING_PATIENCE_STAGE1 = 5
    EARLY_STOPPING_PATIENCE_STAGE2 = 7

    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4

    # Memory / performance knobs
    # Enable gradient checkpointing on transformer backbone to trade compute for memory
    GRADIENT_CHECKPOINTING = True

    # Mixed precision (automatic mixed precision) to save memory and speed up training
    USE_AMP = True

    # Gradient accumulation to achieve larger effective batch sizes with small per-step batches
    GRADIENT_ACCUMULATION_STEPS = 2

    # Output
    OUTPUT_DIR = './outputs'

    # Experiment tracking (optional)
    USE_WANDB = False
    WANDB_PROJECT = 'dfu-pipeline'
    WANDB_ENTITY = None  # set to your wandb username/team for shared projects
    WANDB_RUN_NAME = None