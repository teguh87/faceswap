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
    
    def restore_face(self, face_img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restore and enhance a face image.
        Args:
            face_img (np.ndarray): Full frame image (BGR).
            mask (Optional[np.ndarray]): Optional mask for the face region.
        Returns:
            Tuple[np.ndarray, np.ndarray]: (cropped_restored_face, full_restored_image)
        """
        try:
            # GFPGAN enhance returns (cropped_face, restored_img)
            cropped_face, restored_img = self.gfpganer.enhance(
                face_img, has_aligned=False, only_center_face=False
            )
            
            # If mask is provided, blend only the masked region
            if mask is not None:
                # Create a copy of the original image
                final_img = face_img.copy()
                
                # Normalize mask
                mask_norm = mask.astype(np.float32) / 255.0
                if len(mask_norm.shape) == 2:
                    mask_norm = mask_norm[:, :, np.newaxis]
                
                # Blend restored image with original using mask
                final_img = (restored_img.astype(np.float32) * mask_norm +
                           final_img.astype(np.float32) * (1 - mask_norm))
                final_img = np.clip(final_img, 0, 255).astype(np.uint8)
                
                return cropped_face, final_img
            else:
                return cropped_face, restored_img
                
        except Exception as e:
            print(f"[WARN] GFPGAN face restoration failed: {e}")
            # Return original image for both outputs to maintain tuple format
            return face_img, face_img