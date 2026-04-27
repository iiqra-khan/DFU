# Kaggle Setup & Running Commands

## Prerequisites
- GitHub repo pushed with code
- Kaggle notebook created (Python)
- FUSeg and DPM datasets uploaded to Kaggle datasets or your notebook inputs
- GPU enabled in notebook settings

---

## Quick Copy-Paste Commands for Kaggle Notebook

### 1. Initial Setup (Run Once)

```bash
# Clone repository from GitHub
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME/scripts

# Install dependencies
!pip install -r requirements.txt -q

# Verify GPU availability
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### 2. Setup Data Paths (Modify for Your Dataset Locations)

```python
import os
import shutil

# Option A: If datasets are in Kaggle input (preferred)
# Copy to working directory with symbolic links
os.makedirs('/kaggle/working/data', exist_ok=True)

# Adjust these paths to match your Kaggle dataset names
fuseg_src = '/kaggle/input/fuseg-dataset/fuseg'  # Adjust to your dataset
dpm_src = '/kaggle/input/dpm-dataset/dpm_v3'     # Adjust to your dataset

if os.path.exists(fuseg_src):
    os.system(f'ln -s {fuseg_src} /kaggle/working/data/fuseg')
if os.path.exists(dpm_src):
    os.system(f'ln -s {dpm_src} /kaggle/working/data/dpm_v3')

# Update config.py paths
config_path = 'config.py'
with open(config_path, 'r') as f:
    config_content = f.read()

config_content = config_content.replace(
    "FUSEG_PATH = '../data/fuseg'",
    "FUSEG_PATH = '/kaggle/working/data/fuseg'"
)
config_content = config_content.replace(
    "DPM_PATH = '../data/dpm_v3'",
    "DPM_PATH = '/kaggle/working/data/dpm_v3'"
)
config_content = config_content.replace(
    "OUTPUT_DIR = './outputs'",
    "OUTPUT_DIR = '/kaggle/working/outputs'"
)

with open(config_path, 'w') as f:
    f.write(config_content)

print("✅ Paths updated in config.py")
print(f"FUSeg path: {fuseg_src}")
print(f"DPM path: {dpm_src}")
```

### 3. Train Stage 1 (Segmentation)

```bash
# Train SegFormer-B2 on FUSeg for ulcer segmentation
!python train_stage1_segmentation.py

# Expected output: 
# - outputs/stage1_segformer_best.pth (full model)
# - outputs/encoder_pretrained.pth (encoder for Stage 2)
# - Best Dice score should be ≥ 0.75
```

**Expected runtime**: 15-30 minutes (20 epochs on GPU)

### 4. Train Stage 2 with Strong Transfer (Recommended)

```bash
# Option A: Two-stage transfer (recommended - uses Stage 1 encoder)
!python train_stage2_wagner.py --mode two-stage

# Expected output:
# - outputs/best_wagner_model_two_stage.pth
# - Weighted F1 should be ~0.55-0.62
```

**Expected runtime**: 30-50 minutes (50 epochs on GPU)

### 5. Run Full Ablation Study (Compares All 3 Modes)

```bash
# Run all three configurations sequentially:
# 1. Scratch (baseline)
# 2. ImageNet only
# 3. Two-stage transfer
!python train_stage2_wagner.py --mode ablation

# Expected output:
# - outputs/best_wagner_model_scratch.pth
# - outputs/best_wagner_model_imagenet.pth
# - outputs/best_wagner_model_two_stage.pth
# - outputs/ablation_results.json (summary)
```

**Expected runtime**: 2.5-3 hours total (3 × 50 epochs)

### 6. View Results

```python
import json
import pandas as pd

# Read ablation results
with open('/kaggle/working/outputs/ablation_results.json', 'r') as f:
    results = json.load(f)

# Display comparison table
df = pd.DataFrame(results).T
print("="*60)
print("ABLATION STUDY RESULTS")
print("="*60)
print(df.to_string())
print("\nImprovement (Two-Stage vs Scratch):")
improvement = ((results['two_stage']['best_f1_weighted'] - 
                results['scratch']['best_f1_weighted']) / 
               results['scratch']['best_f1_weighted'] * 100)
print(f"  +{improvement:.1f}% F1 weighted improvement")
```

### 7. Save Outputs to Kaggle

```bash
# Create archive of all outputs
!cd /kaggle/working && tar -czf dfu-pipeline-outputs.tar.gz outputs/
!ls -lh /kaggle/working/dfu-pipeline-outputs.tar.gz

print("✅ Outputs saved and ready to download")
print("   Location: /kaggle/working/dfu-pipeline-outputs.tar.gz")
```

---

## Complete Kaggle Notebook Script (Copy This)

For a full automated notebook, use this Python script:

```python
# ============================================================================
# Kaggle Notebook: DFU Two-Stage Pipeline Training
# ============================================================================

import os
import sys
import json
import subprocess

# Step 1: Setup
print("=" * 70)
print("STEP 1: CLONING REPOSITORY")
print("=" * 70)

# Clone repo
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
os.chdir('YOUR_REPO_NAME/scripts')
print("✅ Repository cloned")

# Install dependencies
print("\nInstalling dependencies...")
!pip install -r requirements.txt -q
print("✅ Dependencies installed")

# Verify GPU
print("\nGPU Status:")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Step 2: Setup Paths
print("\n" + "=" * 70)
print("STEP 2: CONFIGURING PATHS")
print("=" * 70)

os.makedirs('/kaggle/working/data', exist_ok=True)

# ADJUST THESE PATHS TO YOUR KAGGLE DATASETS
fuseg_src = '/kaggle/input/your-fuseg-dataset/fuseg'
dpm_src = '/kaggle/input/your-dpm-dataset/dpm_v3'

# Create symbolic links
if os.path.exists(fuseg_src):
    os.system(f'ln -s {fuseg_src} /kaggle/working/data/fuseg')
    print(f"✅ FUSeg linked: {fuseg_src}")
else:
    print(f"⚠️  FUSeg not found at: {fuseg_src}")

if os.path.exists(dpm_src):
    os.system(f'ln -s {dpm_src} /kaggle/working/data/dpm_v3')
    print(f"✅ DPM linked: {dpm_src}")
else:
    print(f"⚠️  DPM not found at: {dpm_src}")

# Update config paths
config_path = 'config.py'
with open(config_path, 'r') as f:
    config_content = f.read()

config_content = config_content.replace(
    "FUSEG_PATH = '../data/fuseg'",
    "FUSEG_PATH = '/kaggle/working/data/fuseg'"
)
config_content = config_content.replace(
    "DPM_PATH = '../data/dpm_v3'",
    "DPM_PATH = '/kaggle/working/data/dpm_v3'"
)
config_content = config_content.replace(
    "OUTPUT_DIR = './outputs'",
    "OUTPUT_DIR = '/kaggle/working/outputs'"
)

with open(config_path, 'w') as f:
    f.write(config_content)

print("✅ Config paths updated")

# Step 3: Train Stage 1
print("\n" + "=" * 70)
print("STEP 3: TRAINING STAGE 1 (SEGMENTATION)")
print("=" * 70)
print("Model: SegFormer-B2")
print("Dataset: FUSeg")
print("Task: Ulcer segmentation")
print("Expected Dice: ≥ 0.75")
print("-" * 70)

os.system('python train_stage1_segmentation.py')

# Step 4: Train Stage 2
print("\n" + "=" * 70)
print("STEP 4: TRAINING STAGE 2 (WAGNER GRADING)")
print("=" * 70)
print("Mode: Two-stage transfer (Stage 1 encoder → Stage 2 classifier)")
print("Expected F1: 0.55-0.62")
print("-" * 70)

os.system('python train_stage2_wagner.py --mode two-stage')

# Step 5: Run Ablation (Optional - Uncomment to Run)
print("\n" + "=" * 70)
print("STEP 5: ABLATION STUDY (Optional)")
print("=" * 70)
print("Comparing 3 configurations: scratch, imagenet, two-stage")
print("Runtime: ~2.5-3 hours")
print("-" * 70)

# Uncomment the line below to run ablation
# os.system('python train_stage2_wagner.py --mode ablation')

# Step 6: Results
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

import glob
output_dir = '/kaggle/working/outputs'

# List checkpoints
checkpoints = glob.glob(f'{output_dir}/*.pth')
print("\nCheckpoints saved:")
for ckpt in sorted(checkpoints):
    size_mb = os.path.getsize(ckpt) / (1024**2)
    print(f"  - {os.path.basename(ckpt)} ({size_mb:.1f} MB)")

# Show ablation results if available
ablation_file = f'{output_dir}/ablation_results.json'
if os.path.exists(ablation_file):
    with open(ablation_file, 'r') as f:
        results = json.load(f)
    
    print("\nAblation Study Results:")
    print("-" * 40)
    for mode, metrics in results.items():
        print(f"  {mode:>12}: F1_w={metrics['best_f1_weighted']:.4f}, "
              f"F1_m={metrics['best_f1_macro']:.4f}")

print("\n✅ Training complete!")
print(f"   All outputs saved to: {output_dir}")
```

---

## Individual Commands (Pick What You Need)

### Just Stage 1
```bash
!pip install -r requirements.txt -q
!python train_stage1_segmentation.py
```

### Just Stage 2 (Two-Stage Transfer)
```bash
# Assumes encoder_pretrained.pth from Stage 1 already exists
!python train_stage2_wagner.py --mode two-stage
```

### Just Ablation Study
```bash
# Runs all 3 modes sequentially
!python train_stage2_wagner.py --mode ablation
```

### Individual Ablation Modes
```bash
# Scratch (no pretrain)
!python train_stage2_wagner.py --mode scratch

# ImageNet only (no Stage 1)
!python train_stage2_wagner.py --mode imagenet

# Two-stage (uses Stage 1)
!python train_stage2_wagner.py --mode two-stage
```

---

## Kaggle Dataset Linking

### Option 1: Add Datasets to Notebook (GUI)
1. Click "Input" (folder icon) in Kaggle notebook
2. Click "+ Add input"
3. Search for your FUSeg and DPM datasets
4. Datasets appear in `/kaggle/input/dataset-name/`

### Option 2: Find Correct Paths

```python
import os

# List all input datasets
input_dir = '/kaggle/input'
datasets = os.listdir(input_dir)
print("Available datasets:")
for ds in datasets:
    ds_path = os.path.join(input_dir, ds)
    contents = os.listdir(ds_path)
    print(f"\n{ds}:")
    for item in contents[:5]:  # Show first 5 items
        print(f"  - {item}")
```

Then update the paths in the setup section accordingly.

---

## Troubleshooting

### "Module not found" errors
```bash
# Reinstall with explicit upgrade
!pip install --upgrade torch torchvision transformers -q
```

### Out of memory (OOM) errors
```python
# Reduce batch size in config.py
BATCH_SIZE = 8  # Instead of 16

# Or reduce number of workers
NUM_WORKERS = 0  # Instead of 4
```

### Datasets not found
```python
# Debug: Check what's in /kaggle/input
import os
print(os.listdir('/kaggle/input'))

# And working directory
print(os.listdir('/kaggle/working'))
```

### Slow I/O
```bash
# Copy data locally for faster access (if space allows)
!cp -r /kaggle/input/fuseg-dataset/fuseg /kaggle/working/data/fuseg
!cp -r /kaggle/input/dpm-dataset/dpm_v3 /kaggle/working/data/dpm_v3
```

---

## Example: Expected Output in Kaggle

After running the pipeline, your outputs should look like:

```
/kaggle/working/outputs/
├── stage1_segformer_best.pth         (Stage 1 model)
├── encoder_pretrained.pth            (Encoder for Stage 2)
├── best_wagner_model_two_stage.pth   (Stage 2 best model)
├── best_wagner_model_scratch.pth     (Ablation: scratch)
├── best_wagner_model_imagenet.pth    (Ablation: ImageNet)
└── ablation_results.json             (Summary table)

Example ablation_results.json:
{
  "scratch": {
    "best_f1_weighted": 0.452,
    "best_f1_macro": 0.389
  },
  "imagenet": {
    "best_f1_weighted": 0.523,
    "best_f1_macro": 0.451
  },
  "two_stage": {
    "best_f1_weighted": 0.612,
    "best_f1_macro": 0.523
  }
}
```

---

## Tips for Kaggle

1. **Enable GPU**: Settings → Accelerator → GPU (P100 or better)
2. **Set Notebook Timeout**: Settings → Notebook Timeout → Highest (36 hours)
3. **Use Kaggle Secrets**: For private GitHub repos, add GitHub token as secret
4. **Monitor RAM**: Use `!free -h` to check memory during training
5. **Commit Often**: Use `!git push` to save checkpoints back to GitHub
6. **Download Results**: After training, download the `.tar.gz` file

---

## Quick Reference: Runtime Expectations

| Stage | Time (GPU) | Model | Output |
|-------|-----------|-------|--------|
| Stage 1 | 15-30 min | SegFormer-B2 | encoder_pretrained.pth |
| Stage 2 (two-stage) | 30-50 min | SegFormer classifier | best_wagner_model_two_stage.pth |
| Ablation (all 3) | 2.5-3 hrs | 3× Stage 2 | ablation_results.json |
| **Total** | **~1 hour** | **Both stages** | **All checkpoints** |

---

## Next: Download & Analyze Results

```python
# View best metrics
import json
with open('/kaggle/working/outputs/ablation_results.json') as f:
    results = json.load(f)
    print(json.dumps(results, indent=2))

# Download everything
!tar -czf /kaggle/working/outputs.tar.gz /kaggle/working/outputs/
print("✅ Ready to download: outputs.tar.gz")
```
