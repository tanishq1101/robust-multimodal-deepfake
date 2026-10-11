"""Gradio demo launcher (Phase 7).

Upload a video -> overall real/fake verdict, face and voice scores, Grad-CAM heatmaps.

Usage:
  python app/app.py                                   # best checkpoint in <data_root>/checkpoints
  python app/app.py --checkpoint path/to/best.pt
  python app/app.py --share                           # public link (use this on Colab)
"""

import argparse
import sys

from deepfake.demo import create_demo
from deepfake.inference import DeepfakeDetector, find_best_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device")
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)

    ckpt = args.checkpoint or find_best_checkpoint()
    if not ckpt:
        sys.exit("No checkpoint found. Train a model first (notebooks 03-05) or pass --checkpoint.")
    print(f"Loading {ckpt}")
    demo = create_demo(DeepfakeDetector(ckpt, args.device, args.threshold))
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
