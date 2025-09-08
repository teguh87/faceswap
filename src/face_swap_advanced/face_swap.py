"""
Face Swap Advanced - Core face swapping functionality
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
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'FaceSwapConfig':
        """Create configuration from command line arguments"""
        providers_override = FaceSwapUtils.parse_providers_override(args.providers)
        return cls(
            src_img_path=args.src,
            ref_img_path=args.ref,
            tgt_path=args.tgt,
            model_path=args.model,
            out_dir=args.out,
            debug=args.debug,
            skip=args.skip,
            batch_size=args.batch,
            device=args.device,
            providers_override=providers_override,
            min_similarity=args.min_similarity,
            max_face_area_ratio=args.max_face_ratio,
            min_face_size=args.min_face_size
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
        """
        Parse device string and return (ctx_id, providers)
        - ctx_id for FaceAnalysis.prepare: >=0 for CUDA device index, -1 for CPU
        - providers list for ONNX Runtime models (INSwapper and (internally) FaceAnalysis)
        """
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


class FaceSwapper:
    """Main face swap class"""
    
    @staticmethod
    def face_swap(config: FaceSwapConfig) -> str:
        """
        Static method to perform face swapping on images or videos
        
        Args:
            config: FaceSwapConfig object containing all parameters
            
        Returns:
            str: Path to the output file
        """
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

        similarities = [FaceSwapUtils.cosine_similarity(ref_face.embedding, f.embedding) for f in tgt_faces]
        best_idx = int(np.argmax(similarities))
        best_face = tgt_faces[best_idx]

        result_img = swapper.get(tgt_img.copy(), best_face, src_face, paste_back=True)

        out_file = out_dir / FaceSwapUtils.random_name(8, ".png")
        cv2.imwrite(str(out_file), result_img)
        print(f"[INFO] Saved swapped image: {out_file}")

        if config.debug:
            FaceSwapper._save_debug_image(tgt_img, tgt_faces, best_idx, out_dir)

        return str(out_file)

    @staticmethod
    def _process_video(config: FaceSwapConfig, app: FaceAnalysis, swapper: INSwapper, 
                      src_face, ref_face, out_dir: Path) -> str:
        """Process a video file"""
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

        frame_idx = 0
        frames_buffer = []
        processed_frames = 0

        with tqdm(total=total_frames, desc="Swapping faces", unit="frame") as pbar:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1

                    if config.skip > 1 and (frame_idx % config.skip != 0):
                        writer.write(frame)
                        pbar.update(1)
                        continue

                    frames_buffer.append(frame)

                    # Process in batches
                    if len(frames_buffer) >= config.batch_size:
                        FaceSwapper._process_frame_batch(frames_buffer, app, swapper, src_face, ref_face, writer, pbar)
                        frames_buffer = []
                        processed_frames += config.batch_size

                # Process leftover frames
                if frames_buffer:
                    FaceSwapper._process_frame_batch(frames_buffer, app, swapper, src_face, ref_face, writer, pbar)
                    processed_frames += len(frames_buffer)

            finally:
                cap.release()
                writer.release()

        print(f"[INFO] Saved swapped video: {out_file}")
        return str(out_file)

    @staticmethod
    def _process_frame_batch(frames_buffer: List[np.ndarray], app: FaceAnalysis, swapper: INSwapper,
                           src_face, ref_face, writer: cv2.VideoWriter, pbar: tqdm) -> None:
        """Process a batch of video frames"""
        faces_batch = [app.get(f) for f in frames_buffer]
        for frame, faces in zip(frames_buffer, faces_batch):
            if not faces:
                writer.write(frame)
                pbar.update(1)
                continue

            similarities = [FaceSwapUtils.cosine_similarity(ref_face.embedding, f.embedding) for f in faces]
            best_idx = int(np.argmax(similarities))
            best_face = faces[best_idx]

            result_frame = swapper.get(frame.copy(), best_face, src_face, paste_back=True)
            writer.write(result_frame)
            pbar.update(1)

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
    parser = argparse.ArgumentParser(description="Face Swap (Image/Video) with selectable execution provider + batching")
    parser.add_argument("--src", required=True, help="Path to source face image.")
    parser.add_argument("--ref", required=True, help="Path to reference face image.")
    parser.add_argument("--tgt", required=True, help="Path to target image or video.")
    parser.add_argument("--model", required=True, help="Path to inswapper_128.onnx model.")
    parser.add_argument("--out", default="output", help="Directory to save results.")
    parser.add_argument("--debug", action="store_true", help="Overlay debug info (face boxes & IDs).")
    parser.add_argument("--skip", type=int, default=1, help="Frame skip factor for video (1 = no skip).")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for video frame processing.")
    parser.add_argument("--device", default="auto",
                        help="Execution device: auto | cpu | cuda[:index]. Examples: cpu, cuda, cuda:0, cuda:1")
    parser.add_argument("--providers", default=None,
                        help="Override providers list (comma-separated). Example: 'CUDAExecutionProvider,CPUExecutionProvider'")
    return parser


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = FaceSwapConfig.from_args(args)
    
    # Perform face swap using the static method
    result_path = FaceSwapper.face_swap(config)
    print(f"[SUCCESS] Face swap completed. Output saved to: {result_path}")