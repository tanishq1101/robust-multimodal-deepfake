"""Gradio demo launcher.

Video, audio and image analysis with verdicts, per-model scores, timelines and Grad-CAM evidence,
labelled demo samples, and the loaded models' training data, metrics and code.

Usage:
  python scripts/download_model.py                    # once: voice model + pretrained face model
  python app/app.py                                   # best checkpoint (data checkpoints, then models/)
  python app/app.py --checkpoint path/to/best.pt
  python app/app.py --share                           # public link (use this on Colab)
"""

import argparse
import sys

from deepfake.analyzer import Analyzer
from deepfake.demo import create_demo, launch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="project checkpoint (default: best available)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device")
    p.add_argument("--no-face", action="store_true", help="do not load the pretrained face model")
    p.add_argument("--share", action="store_true", help="create a public gradio.live link")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)

    try:
        analyzer = Analyzer.from_defaults(args.checkpoint, args.device, args.threshold, face=not args.no_face)
    except ValueError:
        sys.exit("No model found. Download the published models with `python scripts/download_model.py`, "
                 "train one (notebooks/deepfake_pipeline.ipynb), or pass --checkpoint.")
    caps = analyzer.capabilities()
    print(f"Face model : {caps['face_model'] or 'not loaded (python scripts/download_model.py face_dfdc_effb7)'}")
    print(f"Voice model: {caps['voice_model'] or 'not loaded (python scripts/download_model.py)'}")
    launch(create_demo(analyzer), share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
