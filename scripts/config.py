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

    # Single-task SegFormer-B2 baseline for paper comparison
    BASELINE_BACKBONE = 'nvidia/mit-b2'
    BASELINE_NUM_CLASSES = 4
    BASELINE_IMG_SIZE = 224
    BASELINE_BATCH_SIZE = 16
    BASELINE_EPOCHS = 30
    BASELINE_LR = 1e-4
    BASELINE_NUM_RUNS = 3
    BASELINE_SEED = 42
    BASELINE_OUTPUT_DIR = './outputs/baseline_segformer_b2'
    
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
    NUM_WORKERS = 2

    # Memory / performance knobs
    # Enable gradient checkpointing on transformer backbone to trade compute for memory
    GRADIENT_CHECKPOINTING = True

    # Mixed precision (automatic mixed precision) to save memory and speed up training
    USE_AMP = True

    # Gradient accumulation to achieve larger effective batch sizes with small per-step batches
    GRADIENT_ACCUMULATION_STEPS = 2

    # Validate less often to reduce Kaggle runtime overhead.
    # Validation still runs on the final epoch.
    VALIDATE_EVERY_N_EPOCHS = 2

    # Persist metrics/figures for notebook review and paper figures
    SAVE_METRICS_JSON = True
    SAVE_PLOTS = True

    # Output
    OUTPUT_DIR = './outputs'

    # Experiment tracking (optional)
    USE_WANDB = False
    WANDB_PROJECT = 'dfu-pipeline'
    WANDB_ENTITY = None  # set to your wandb username/team for shared projects
    WANDB_RUN_NAME = None
