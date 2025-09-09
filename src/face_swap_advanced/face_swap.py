#!/usr/bin/env python3
"""
Face Swap Advanced - Standalone Script
This is a complete standalone version that can be run without package installation.
"""

from tqdm import tqdm
import cv2
import numpy as np
import random
import string
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

from .face_restorer import FaceRestorer


@dataclass
class FaceSwapConfig:
    """Configuration class for face swap parameters"""
    src_img_path: str
    ref_img_path: str
    tgt_path: str
    model_path: str
    out_dir: str = "output"
    debug: bool = False
    skip: int = 1
    batch_size: int = 8
    device: str = "auto"
    providers_override: Optional[List[str]] = None
    min_similarity: float = 0.3  # Minimum similarity threshold
    max_face_area_ratio: float = 0.8  # Maximum face area as ratio of frame area
    min_face_size: int = 50  # Minimum face size in pixels

    # ✅ Add GFPGAN attributes
    gfpgan_model: Optional[str] = None
    gfpgan_upscale: int = 1

    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'FaceSwapConfig':
        """Create configuration from command line arguments"""
        providers_override = FaceSwapUtils.parse_providers_override(getattr(args, 'providers', None))
        
        # Handle missing attributes gracefully with defaults
        return cls(
            src_img_path=args.src,
            ref_img_path=args.ref,
            tgt_path=args.tgt,
            model_path=args.model,
            out_dir=getattr(args, 'out', 'output'),
            debug=getattr(args, 'debug', False),
            skip=getattr(args, 'skip', 1),
            batch_size=getattr(args, 'batch', 8),
            device=getattr(args, 'device', 'auto'),
            providers_override=providers_override,
            min_similarity=getattr(args, 'min_similarity', 0.3),
            max_face_area_ratio=getattr(args, 'max_face_ratio', 0.8),
            min_face_size=getattr(args, 'min_face_size', 50),
            gfpgan_model=getattr(args, 'gfpgan_model', None),
            gfpgan_upscale=getattr(args, 'gfpgan_upscale', 1)
        )


class FaceSwapUtils:
    """Utility methods for face swap operations"""
    
    @staticmethod
    def random_name(length: int = 8, ext: str = ".png") -> str:
        """Generate a random filename"""
        letters = string.ascii_lowercase + string.digits
        return ''.join(random.choice(letters) for _ in range(length)) + ext

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return float(np.dot(a_norm, b_norm))

    @staticmethod
    def parse_device(device: str) -> tuple[int, List[str]]:
        """Parse device string and return (ctx_id, providers)"""
        device = (device or "auto").lower().strip()

        if device == "cpu":
            return -1, ["CPUExecutionProvider"]

        if device.startswith("cuda"):
            idx = 0
            if ":" in device:
                try:
                    idx = int(device.split(":", 1)[1])
                except ValueError:
                    idx = 0
            return idx, ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # auto: prefer CUDA:0, fallback to CPU if unavailable
        return 0, ["CUDAExecutionProvider", "CPUExecutionProvider"]

    @staticmethod
    def parse_providers_override(s: Optional[str]) -> Optional[List[str]]:
        """Parse providers override string"""
        if not s:
            return None
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return parts or None

    @staticmethod
    def ensure_available_providers(requested_providers: List[str]) -> List[str]:
        """Ensure requested providers are available"""
        available_providers = ort.get_available_providers()
        selected = [p for p in requested_providers if p in available_providers]
        if not selected:
            print(f"[WARN] None of the requested providers are available: {requested_providers}")
            print(f"[INFO] Falling back to available provider(s): {available_providers}")
            selected = available_providers
        return selected

    @staticmethod
    def is_face_valid(face, frame_shape: tuple, config: FaceSwapConfig) -> bool:
        """Check if a detected face is valid (not anomalous)"""
        if face is None:
            return False
            
        # Check face bounding box
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Check minimum face size
        if face_width < config.min_face_size or face_height < config.min_face_size:
            return False
        
        # Check if face is within frame bounds
        frame_height, frame_width = frame_shape[:2]
        if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
            return False
        
        # Check face area ratio (avoid faces that are too large - might be false positives)
        face_area = face_width * face_height
        frame_area = frame_height * frame_width
        face_area_ratio = face_area / frame_area
        
        if face_area_ratio > config.max_face_area_ratio:
            return False
        
        # Check face confidence if available
        if hasattr(face, 'det_score') and face.det_score < 0.5:
            return False
            
        return True

    @staticmethod
    def find_best_matching_face(faces, ref_face, frame_shape: tuple, config: FaceSwapConfig):
        """Find the best matching face from detected faces, filtering out anomalies"""
        if not faces:
            return None, 0
        
        # Filter valid faces first
        valid_faces = [(face, i) for i, face in enumerate(faces) 
                      if FaceSwapUtils.is_face_valid(face, frame_shape, config)]
        
        if not valid_faces:
            return None, 0
        
        # Calculate similarities for valid faces
        similarities = []
        for face, original_idx in valid_faces:
            similarity = FaceSwapUtils.cosine_similarity(ref_face.embedding, face.embedding)
            similarities.append((similarity, face, original_idx))
        
        # Sort by similarity and get the best one
        similarities.sort(key=lambda x: x[0], reverse=True)
        best_similarity, best_face, best_idx = similarities[0]
        
        # Check if similarity meets minimum threshold
        if best_similarity < config.min_similarity:
            return None, 0
            
        return best_face, best_similarity


class FaceSwapper:
    """Main face swap class"""

    # Add GFPGAN restorer to swapper
    gfpgan_restorer: Optional[FaceRestorer] = None

    @staticmethod
    def set_restorer(restorer: Optional[FaceRestorer]):
        """Set the GFPGAN restorer"""
        FaceSwapper.gfpgan_restorer = restorer
    
    @staticmethod
    def face_swap(config: FaceSwapConfig) -> str:
        """Static method to perform face swapping on images or videos"""
        # Device / providers
        ctx_id, providers = FaceSwapUtils.parse_device(config.device)
        if config.providers_override:
            providers = config.providers_override
        providers = FaceSwapUtils.ensure_available_providers(providers)
        print(f"[INFO] Using providers: {providers} | FaceAnalysis ctx_id={ctx_id}")

        # InsightFace FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        # INSwapper (ONNXRuntime under the hood)
        swapper = INSwapper(config.model_path)
        swapper.session = ort.InferenceSession(config.model_path, providers=providers)
        print(f"[INFO] INSwapper session using providers: {swapper.session.get_providers()}")

        pad_size = 150
        out_dir = Path(config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- Load source face ---
        src_img = cv2.imread(str(config.src_img_path))
        if src_img is None:
            raise FileNotFoundError(f"Source image not found: {config.src_img_path}")
        src_padded = cv2.copyMakeBorder(src_img, pad_size, pad_size, pad_size, pad_size,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        src_faces = app.get(src_padded)
        if not src_faces:
            raise RuntimeError("No face detected in source image!")
        src_face = src_faces[0]

        # --- Load reference face ---
        ref_img = cv2.imread(str(config.ref_img_path))
        if ref_img is None:
            raise FileNotFoundError(f"Reference image not found: {config.ref_img_path}")
        ref_padded = cv2.copyMakeBorder(ref_img, pad_size, pad_size, pad_size, pad_size,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        ref_faces = app.get(ref_padded)
        if not ref_faces:
            raise RuntimeError("No face detected in reference image!")
        ref_face = ref_faces[0]

        # --- Detect if target is image or video ---
        tgt_path = Path(config.tgt_path)
        is_video = tgt_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        if not is_video:
            return FaceSwapper._process_image(config, app, swapper, src_face, ref_face, out_dir)
        else:
            return FaceSwapper._process_video(config, app, swapper, src_face, ref_face, out_dir)

    @staticmethod
    def _process_image(config: FaceSwapConfig, app: FaceAnalysis, swapper: INSwapper, 
                      src_face, ref_face, out_dir: Path) -> str:
        """Process a single image"""
        tgt_img = cv2.imread(str(config.tgt_path))
        if tgt_img is None:
            raise FileNotFoundError(f"Target image not found: {config.tgt_path}")

        tgt_faces = app.get(tgt_img)
        if not tgt_faces:
            raise RuntimeError("No faces detected in target image!")

        # Use the enhanced face matching with anomaly detection
        best_face, similarity = FaceSwapUtils.find_best_matching_face(
            tgt_faces, ref_face, tgt_img.shape, config
        )
        
        if best_face is None:
            raise RuntimeError(f"No suitable face found in target image! "
                             f"Minimum similarity threshold: {config.min_similarity}")
        """
        result_img = swapper.get(tgt_img.copy(), best_face, src_face, paste_back=True)

        out_file = out_dir / FaceSwapUtils.random_name(8, ".png")
        cv2.imwrite(str(out_file), result_img)
        print(f"[INFO] Saved swapped image: {out_file} (similarity: {similarity:.3f})")

        if config.debug:
            FaceSwapper._save_debug_image(tgt_img, tgt_faces, -1, out_dir)

        return str(out_file)    
        """
        swapped_face = swapper.get(tgt_img.copy(), best_face, src_face, paste_back=False)

        # Apply GFPGAN if available
        if FaceSwapper.gfpgan_restorer:
            print("[INFO] face restore: ")
            # Generate face mask
            mask = np.zeros(tgt_img.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = best_face.bbox.astype(int)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            swapped_face = FaceSwapper.gfpgan_restorer.restore_face(swapped_face, mask)

        # Paste back using original swapper method
        final_result = swapper.get(tgt_img.copy(), best_face, src_face, paste_back=True)
        
        out_file = out_dir / FaceSwapUtils.random_name(8, ".png")
        cv2.imwrite(str(out_file), final_result)
        print(f"[INFO] Saved swapped image: {out_file} (similarity: {similarity:.3f})")
        return str(out_file)
        

    @staticmethod
    def _process_video(config: FaceSwapConfig, app: FaceAnalysis, swapper: INSwapper, 
                      src_face, ref_face, out_dir: Path) -> str:
        """Process a video file with anomaly detection and similarity filtering"""
        cap = cv2.VideoCapture(str(config.tgt_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {config.tgt_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        out_file = out_dir / FaceSwapUtils.random_name(8, ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(f"[INFO] Total frames in video: {total_frames}")
        print(f"[INFO] Similarity threshold: {config.min_similarity}")

        frame_idx = 0
        frames_buffer = []
        processed_frames = 0
        skipped_frames = 0
        swapped_frames = 0

        with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1

                    # Skip frames based on skip parameter
                    if config.skip > 1 and (frame_idx % config.skip != 0):
                        writer.write(frame)
                        pbar.update(1)
                        continue

                    frames_buffer.append((frame, frame_idx))

                    # Process in batches
                    if len(frames_buffer) >= config.batch_size:
                        batch_stats = FaceSwapper._process_frame_batch_with_filtering(
                            frames_buffer, app, swapper, src_face, ref_face, writer, pbar, config
                        )
                        processed_frames += len(frames_buffer)
                        skipped_frames += batch_stats['skipped']
                        swapped_frames += batch_stats['swapped']
                        frames_buffer = []

                # Process leftover frames
                if frames_buffer:
                    
                    batch_stats = FaceSwapper._process_frame_batch_with_filtering(
                        frames_buffer, app, swapper, src_face, ref_face, writer, pbar, config
                    )
                    processed_frames += len(frames_buffer)
                    skipped_frames += batch_stats['skipped']
                    swapped_frames += batch_stats['swapped']

            finally:
                cap.release()
                writer.release()

        print(f"[INFO] Video processing completed:")
        print(f"[INFO] - Total frames processed: {processed_frames}")
        print(f"[INFO] - Frames with face swapping: {swapped_frames}")
        print(f"[INFO] - Frames skipped (low similarity/anomalies): {skipped_frames}")
        print(f"[INFO] - Success rate: {(swapped_frames/processed_frames)*100:.1f}%")
        print(f"[INFO] Saved swapped video: {out_file}")
        return str(out_file)

    @staticmethod
    def _process_frame_batch_with_filtering(frames_data: List[tuple], app: FaceAnalysis, swapper: INSwapper,
                                          src_face, ref_face, writer: cv2.VideoWriter, pbar: tqdm, 
                                          config: FaceSwapConfig) -> dict:
        """Process a batch of video frames with anomaly detection and similarity filtering"""
        frames = [frame_data[0] for frame_data in frames_data]
        frame_indices = [frame_data[1] for frame_data in frames_data]
        
        # Detect faces in batch
        faces_batch = [app.get(f) for f in frames]
        
        stats = {'skipped': 0, 'swapped': 0}
        
        for i, (frame, faces, frame_idx) in enumerate(zip(frames, faces_batch, frame_indices)):
            try:
                if not faces:
                    # No faces detected - write original frame
                    writer.write(frame)
                    stats['skipped'] += 1
                    if config.debug:
                        print(f"[DEBUG] Frame {frame_idx}: No faces detected")
                else:
                    # Find best matching valid face
                    best_face, similarity = FaceSwapUtils.find_best_matching_face(
                        faces, ref_face, frame.shape, config
                    )
                    
                    if best_face is None:
                        # No suitable face found - write original frame
                        writer.write(frame)
                        stats['skipped'] += 1
                        if config.debug:
                            print(f"[DEBUG] Frame {frame_idx}: No suitable face found (low similarity or anomaly)")
                    else:
                        # Perform face swap
                        swapped_frame = swapper.get(frame.copy(), best_face, src_face, paste_back=True)
                        """
                        writer.write(result_frame)
                        stats['swapped'] += 1
                        if config.debug:
                            print(f"[DEBUG] Frame {frame_idx}: Face swapped (similarity: {similarity:.3f})")
                        """
                        # Step 2: restore with GFPGAN if available
                        if FaceSwapper.gfpgan_restorer:
                            # print(f"[INFO] Restoring frame {frame_idx} face with GFPGAN...")
                            # Create face mask with padding
                            pad = 15
                            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                            x1, y1, x2, y2 = best_face.bbox.astype(int)
                            cv2.rectangle(mask, (x1-pad, y1-pad), (x2+pad, y2+pad), 255, -1)

                            # Apply GFPGAN to the swapped frame
                            # returns: restored_face, restored_face_full
                            restored_frame, _ = FaceSwapper.gfpgan_restorer.restore_face(swapped_frame, mask)

                        else:
                            restored_frame = swapped_frame

                        # Step 3: write the frame
                        writer.write(restored_frame)
                        stats['swapped'] += 1
            except Exception as e:
                # Handle any errors during processing - write original frame
                print(f"[WARN] Error processing frame {frame_idx}: {e}")
                writer.write(frame)
                stats['skipped'] += 1
            
            pbar.update(1)
                
        return stats

    @staticmethod
    def _save_debug_image(tgt_img: np.ndarray, tgt_faces: List, best_idx: int, out_dir: Path) -> None:
        """Save debug image with face detection boxes"""
        vis = tgt_img.copy()
        for i, f in enumerate(tgt_faces):
            x1, y1, x2, y2 = f.bbox.astype(int)
            color = (0, 0, 255) if i == best_idx else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, str(i), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        debug_file = out_dir / FaceSwapUtils.random_name(8, ".png")
        cv2.imwrite(str(debug_file), vis)
        print(f"[INFO] Saved debug image: {debug_file}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(description="Face Swap (Image/Video) with anomaly detection and similarity filtering")
    parser.add_argument("--src", required=True, help="Path to source face image.")
    parser.add_argument("--ref", required=True, help="Path to reference face image.")
    parser.add_argument("--tgt", required=True, help="Path to target image or video.")
    parser.add_argument("--model", required=True, help="Path to inswapper_128.onnx model.")
    parser.add_argument("--out", default="output", help="Directory to save results.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output and save debug info.")
    parser.add_argument("--skip", type=int, default=1, help="Frame skip factor for video (1 = no skip).")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for video frame processing.")
    parser.add_argument("--device", default="auto",
                        help="Execution device: auto | cpu | cuda[:index]. Examples: cpu, cuda, cuda:0, cuda:1")
    parser.add_argument("--providers", default=None,
                        help="Override providers list (comma-separated). Example: 'CUDAExecutionProvider,CPUExecutionProvider'")
    parser.add_argument("--min-similarity", type=float, default=0.3, dest="min_similarity",
                        help="Minimum cosine similarity threshold for face matching (0.0-1.0)")
    parser.add_argument("--max-face-ratio", type=float, default=0.8, dest="max_face_ratio", 
                        help="Maximum face area as ratio of frame area (0.0-1.0)")
    parser.add_argument("--min-face-size", type=int, default=50, dest="min_face_size",
                        help="Minimum face size in pixels")
    parser.add_argument("--gfpgan-model", default=None,
                        help="Path to GFPGAN model (.pth). If provided, restores swapped faces to reduce artifacts.")
    parser.add_argument("--gfpgan-upscale", type=int, default=1,
                        help="GFPGAN upscale factor (1=no upscale, 2=2x, 4=4x)")
    return parser


def main():
    """Main entry point for the standalone script"""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Validate input paths
        if not Path(args.src).exists():
            print(f"[ERROR] Source image not found: {args.src}")
            return 1
            
        if not Path(args.ref).exists():
            print(f"[ERROR] Reference image not found: {args.ref}")
            return 1
            
        if not Path(args.tgt).exists():
            print(f"[ERROR] Target file not found: {args.tgt}")
            return 1
            
        if not Path(args.model).exists():
            print(f"[ERROR] Model file not found: {args.model}")
            return 1
        
        # Create configuration from arguments
        try:
            config = FaceSwapConfig.from_args(args)
        except Exception as e:
            print(f"[ERROR] Configuration error: {e}")
            return 1
        
        # Print configuration summary
        print("[INFO] Face Swap Advanced - Starting Processing")
        print(f"[INFO] Source: {config.src_img_path}")
        print(f"[INFO] Reference: {config.ref_img_path}")
        print(f"[INFO] Target: {config.tgt_path}")
        print(f"[INFO] Model: {config.model_path}")
        print(f"[INFO] Output directory: {config.out_dir}")
        print(f"[INFO] Device: {config.device}")
        print(f"[INFO] GFGAN model: {config.gfpgan_model}")
        print(f"[INFO] Upscale: {config.gfpgan_upscale}")
        print(f"[INFO] Minimum similarity threshold: {config.min_similarity}")
        print(f"[INFO] Maximum face area ratio: {config.max_face_area_ratio}")
        print(f"[INFO] Minimum face size: {config.min_face_size}")
        print(f"[INFO] Batch size: {config.batch_size}")
        

        # 1. Initialize GFPGAN restorer if model path provided
        restorer = FaceRestorer(config.gfpgan_model, upscale=config.gfpgan_upscale) if config.gfpgan_model else None
        FaceSwapper.set_restorer(restorer)

        
        # Perform face swap using the static method
        result_path = FaceSwapper.face_swap(config)
        
        print(f"[SUCCESS] Face swap completed successfully!")
        print(f"[SUCCESS] Output saved to: {result_path}")
        return 0
        
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted by user")
        return 1
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        if hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())