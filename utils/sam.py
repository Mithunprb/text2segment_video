import torch
import numpy as np
from PIL import Image
import supervision as sv  
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

MODEL_CHECKPOINT = "checkpoints/sam2_hiera_small.pt"
MODEL_CONFIG = "sam2_hiera_s.yaml"

def initialize_sam(
    device: torch.device,
    config_path: str = MODEL_CONFIG,
    checkpoint_path: str = MODEL_CHECKPOINT
) -> SAM2ImagePredictor:
    """
    Initializes the SAM2 model.

    Parameters:
        device (torch.device): The device to load the model onto.
        config_path (str): Path to the configuration file.
        checkpoint_path (str): Path to the model checkpoint file.

    Returns:
        SAM2ImagePredictor: The initialized SAM2 image predictor model.
    """
    model = build_sam2(config_path, checkpoint_path, device=device)
    return SAM2ImagePredictor(sam_model=model)

def perform_sam_inference(
    model: SAM2ImagePredictor,
    image: Image.Image,
    detections: sv.Detections
) -> (sv.Detections, float):
    """
    Performs inference using the SAM2 model.

    Parameters:
        model (SAM2ImagePredictor): The initialized SAM2 image predictor model.
        image (PIL.Image.Image): The input image for inference.
        detections (sv.Detections): The detection results to be used for inference.

    Returns:
        sv.Detections: Updated detections with mask.
        float: The prediction score.
    """
    rgb_image = np.array(image.convert("RGB"))
    model.set_image(rgb_image)
    mask, score, _ = model.predict(box=detections.xyxy, multimask_output=False)

    if mask.ndim == 4:
        mask = np.squeeze(mask)

    detections.mask = mask.astype(bool)
    return detections, score
