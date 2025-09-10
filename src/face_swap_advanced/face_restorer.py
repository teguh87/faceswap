from typing import Optional, Tuple
import cv2
import numpy as np

class FaceRestorer:
    """Restore and optionally upscale faces using GFPGAN."""
    
    def __init__(self, model_path: str, upscale: int = 1):
        """
        Initialize GFPGAN model.
        Args:
            model_path (str): Path to GFPGAN .pth model.
            upscale (int): Upscale factor (1=no upscale, 2=2x, 4=4x).
        """
        from gfpgan import GFPGANer
        self.upscale = upscale
        self.gfpganer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        print(f"[INFO] GFPGAN initialized (upscale={upscale})")
    
    def restore_face(self, face_img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            print("[DEBUG] Restoring face with GFPGAN...")
            outputs = self.gfpganer.enhance(face_img, has_aligned=False, only_center_face=False)
            print(f"[DEBUG] GFPGAN outputs type: {type(outputs)}, length: {len(outputs)}")

            restored_face = None

            if isinstance(outputs, tuple) and len(outputs) >= 2:
                # Try the last element (full restored image list)
                candidates = outputs[-1]
                if isinstance(candidates, list) and len(candidates) > 0:
                    restored_face = candidates[0]
                    print(f"[DEBUG] Candidate restored face shape: {restored_face.shape}")
            elif isinstance(outputs, np.ndarray):
                restored_face = outputs

            if restored_face is None:
                print("[WARN] Could not extract valid restored face, using original.")
                return face_img

            # Ensure shape matches input
            if restored_face.shape != face_img.shape:
                print(f"[DEBUG] Resizing restored face from {restored_face.shape} to {face_img.shape}")
                restored_face = cv2.resize(restored_face, (face_img.shape[1], face_img.shape[0]))

            # Blend with mask if provided
            if mask is not None and mask.size > 0:
                print(f"[DEBUG] Applying mask with shape: {mask.shape}")
                mask_norm = mask.astype(np.float32) / 255.0
                if len(mask_norm.shape) == 2:
                    mask_norm = mask_norm[:, :, np.newaxis]
                restored_face = restored_face.astype(np.float32) * mask_norm + face_img.astype(np.float32) * (1 - mask_norm)
                restored_face = np.clip(restored_face, 0, 255).astype(np.uint8)
                print("[DEBUG] Mask applied successfully.")

            return restored_face

        except Exception as e:
            print(f"[WARN] GFPGAN face restoration failed: {e}")
            return face_img




