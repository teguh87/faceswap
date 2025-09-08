#!/usr/bin/env python3
"""
Command Line Interface for Face Swap Advanced
"""

import sys
import argparse
from pathlib import Path

from .face_swap import FaceSwapConfig, FaceSwapper, create_argument_parser


def main():
    """Main entry point for the CLI application"""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Validate input paths
        if not Path(args.src).exists():
            print(f"[ERROR] Source image not found: {args.src}", file=sys.stderr)
            sys.exit(1)
            
        if not Path(args.ref).exists():
            print(f"[ERROR] Reference image not found: {args.ref}", file=sys.stderr)
            sys.exit(1)
            
        if not Path(args.tgt).exists():
            print(f"[ERROR] Target file not found: {args.tgt}", file=sys.stderr)
            sys.exit(1)
            
        if not Path(args.model).exists():
            print(f"[ERROR] Model file not found: {args.model}", file=sys.stderr)
            sys.exit(1)
        
        # Create configuration from arguments
        config = FaceSwapConfig.from_args(args)
        
        # Print configuration summary
        print("[INFO] Face Swap Advanced - Starting Processing")
        print(f"[INFO] Source: {config.src_img_path}")
        print(f"[INFO] Reference: {config.ref_img_path}")
        print(f"[INFO] Target: {config.tgt_path}")
        print(f"[INFO] Model: {config.model_path}")
        print(f"[INFO] Output directory: {config.out_dir}")
        print(f"[INFO] Device: {config.device}")
        print(f"[INFO] Minimum similarity threshold: {config.min_similarity}")
        
        # Perform face swap using the static method
        result_path = FaceSwapper.face_swap(config)
        
        print(f"[SUCCESS] Face swap completed successfully!")
        print(f"[SUCCESS] Output saved to: {result_path}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}", file=sys.stderr)
        if args.debug if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()