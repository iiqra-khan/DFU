"""
Test script to verify the pipeline components work together
"""
import os
import sys
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported"""
    try:
        from config import Config
        print("✅ Config imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False

    try:
        from dataset import FUSegDataset, DPMDataset, get_transforms
        print("✅ Dataset imported successfully")
    except Exception as e:
        print(f"❌ Failed to import dataset: {e}")
        return False

    try:
        from train_stage1_segmentation import train_segmentation
        print("✅ Stage 1 training imported successfully")
    except Exception as e:
        print(f"❌ Failed to import stage 1 training: {e}")
        return False

    try:
        from train_stage2_wagner import train_wagner_grading
        print("✅ Stage 2 training imported successfully")
    except Exception as e:
        print(f"❌ Failed to import stage 2 training: {e}")
        return False

    try:
        from xai_analysis import generate_xai_explanations
        print("✅ XAI analysis imported successfully")
    except Exception as e:
        print(f"❌ Failed to import XAI analysis: {e}")
        return False

    return True

def test_config():
    """Test config values"""
    from config import Config

    print("\n📋 Configuration:")
    print(f"  FUSEG_PATH: {Config.FUSEG_PATH}")
    print(f"  DPM_PATH: {Config.DPM_PATH}")
    print(f"  BACKBONE: {Config.BACKBONE}")
    print(f"  BATCH_SIZE: {Config.BATCH_SIZE}")
    print(f"  EPOCHS_STAGE1: {Config.EPOCHS_STAGE1}")
    print(f"  EPOCHS_STAGE2: {Config.EPOCHS_STAGE2}")
    print(f"  DEVICE: {Config.DEVICE}")
    print(f"  OUTPUT_DIR: {Config.OUTPUT_DIR}")

def test_directory_structure():
    """Check if expected directories exist or can be created"""
    from config import Config

    print("\n📁 Directory Structure:")

    # Check if we can create output directory
    output_dir = Path(Config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"  Output directory: {output_dir.absolute()} {'✅' if output_dir.exists() else '❌'}")

    # Check for data directories (will likely not exist yet, but show expected paths)
    fuseg_train_images = Path(Config.FUSEG_PATH) / "train" / "images"
    fuseg_train_labels = Path(Config.FUSEG_PATH) / "train" / "labels"
    dpm_train = Path(Config.DPM_PATH) / "train"

    print(f"  FUSeg train images: {fuseg_train_images} {'📁' if fuseg_train_images.exists() else '⚠️  (expected missing)'}")
    print(f"  FUSeg train labels: {fuseg_train_labels} {'📁' if fuseg_train_labels.exists() else '⚠️  (expected missing)'}")
    print(f"  DPM V3.3 train: {dpm_train} {'📁' if dpm_train.exists() else '⚠️  (expected missing)'}")

if __name__ == "__main__":
    print("🧪 Testing DFU Pipeline Scripts")
    print("=" * 50)

    if test_imports():
        print("\n🎉 All imports successful!")
        test_config()
        test_directory_structure()
        print("\n🚀 Ready to use! Next steps:")
        print("   1. Download FUSeg and DPM V3.3 datasets")
        print("   2. Place them in the expected directories")
        print("   3. Run: python scripts/train_stage1_segmentation.py")
        print("   4. Then run: python scripts/train_stage2_wagner.py")
    else:
        print("\n💥 Some imports failed. Please check the error messages above.")
        sys.exit(1)