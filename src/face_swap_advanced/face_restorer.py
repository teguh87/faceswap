from typing import Optional
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
        """
        Restore and enhance a face image.
        
        Args:
            face_img (np.ndarray): Cropped/swapped face image (BGR).
            mask (Optional[np.ndarray]): Optional mask for the face region.
        
        Returns:
            np.ndarray: Restored/upscaled face image.
        """
        try:
            restored_face, _ = self.gfpganer.enhance(
                face_img, has_aligned=False, only_center_face=False
            )
            # If mask is provided, blend the restored face with the original
            if mask is not None:
                mask_norm = mask.astype(np.float32) / 255.0
                if len(mask_norm.shape) == 2:
                    mask_norm = mask_norm[:, :, np.newaxis]
                restored_face = (restored_face.astype(np.float32) * mask_norm + 
                                 face_img.astype(np.float32) * (1 - mask_norm))
                restored_face = np.clip(restored_face, 0, 255).astype(np.uint8)
            
            return restored_face
        
        except Exception as e:
            print(f"[WARN] GFPGAN face restoration failed: {e}")
            return face_img
