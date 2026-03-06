import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.al_loop import ActiveLearningLoop

def main():
    parser = argparse.ArgumentParser(description="SegAL-VLM-X Training")
    parser.add_argument('--config_dir', type=str, default='configs', help='Path to config directory')
    args = parser.parse_args()
    
    print("Starting SegAL-VLM-X Active Learning Loop...")
    loop = ActiveLearningLoop(config_dir=args.config_dir)
    loop.run()

if __name__ == "__main__":
    main()
