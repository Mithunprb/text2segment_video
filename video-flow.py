import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

# RAFT-specific imports
raft-pytorch = '/kaggle/input/raft-pytorch'
sys.path.append(raft-pytorch)  # Update this path accordingly
from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder
from raft.config import RAFTConfig

# Existing imports
from utils.utils import annotate_image
from utils.florence import florence_load_model, florence_run_inference, \
    TASK_DETAILED_CAPTION, \
    TASK_CAPTION_TO_PHRASE_GROUNDING, TASK_OPEN_VOCABULARY_DETECTION
from utils.modes import OPEN_VOCABULARY_DETECTION, CAPTION_GROUNDING_MASKS
from utils.sam import initialize_sam, perform_sam_inference
import supervision as sv

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# Initialize Florence and SAM models
FLORENCE_MODEL, FLORENCE_PROCESSOR = florence_load_model(device=DEVICE)
SAM_MODEL = initialize_sam(device=DEVICE)

# RAFT Configuration and Model Initialization
def initialize_raft(device):
    config = RAFTConfig(
        dropout=0,
        alternate_corr=False,
        small=False,
        mixed_precision=False
    )
    raft_model = RAFT(config)
    raft_weights_path = '/kaggle/input/raft-pytorch/raft-sintel.pth'  # Update this path accordingly
    ckpt = torch.load(raft_weights_path, map_location=device)
    raft_model.to(device)
    raft_model.load_state_dict(ckpt)
    raft_model.eval()
    return raft_model

RAFT_MODEL = initialize_raft(DEVICE)
print("RAFT model loaded successfully.")

# Function to compute optical flow and extract foreground
def compute_foreground(frame1, frame2, model, device, threshold=2.0):
    # Convert frames to tensors
    image1 = torch.from_numpy(frame1).permute(2, 0, 1).float().to(device)
    image2 = torch.from_numpy(frame2).permute(2, 0, 1).float().to(device)
    image1 = image1.unsqueeze(0)
    image2 = image2.unsqueeze(0)
    
    # Pad images
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    
    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
    
    # Compute flow magnitude
    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
    magnitude = np.linalg.norm(flow, axis=2)
    
    # Threshold to create mask
    mask = (magnitude > threshold).astype(np.uint8) * 255
    
    # Refine mask with morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    # Convert mask to 3 channels
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    # Extract foreground
    foreground = cv2.bitwise_and(frame1, mask_3ch)
    
    return mask, foreground

# Visualization function (optional, for debugging)
def visualize_flow_and_mask(frame1, frame2, flow, mask, foreground):
    flo_img = flow_viz.flow_to_image(flow)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].set_title('Frame 1')
    axes[0].imshow(frame1)
    axes[1].set_title('Frame 2')
    axes[1].imshow(frame2)
    axes[2].set_title('Optical Flow')
    axes[2].imshow(flo_img)
    axes[3].set_title('Foreground Mask')
    axes[3].imshow(mask, cmap='gray')
    plt.show()
    
    plt.figure(figsize=(5,5))
    plt.title('Extracted Foreground')
    plt.imshow(foreground)
    plt.axis('off')
    plt.show()

# Modified process_video function with RAFT-based foreground extraction
def process_video(input_video_path, output_video_path, mode, text_input=None):
    if not os.path.exists(input_video_path):
        print(f"Error: The file {input_video_path} does not exist.")
        return
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    prev_frame = None  # To store the previous frame for optical flow
    with tqdm(total=total_frames, desc="Processing Video", unit="frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if prev_frame is not None:
                # Compute foreground mask and extract foreground using RAFT
                mask, foreground = compute_foreground(prev_frame, frame_rgb, RAFT_MODEL, DEVICE, threshold=2.0)
                
                # Optional: visualize for debugging
                # visualize_flow_and_mask(prev_frame, frame_rgb, flow, mask, foreground)
            else:
                # For the first frame, assume entire frame is background (no foreground)
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                foreground = frame_rgb.copy()

            # Update previous frame
            prev_frame = frame_rgb.copy()

            # Convert foreground to PIL Image for processing
            image_input = Image.fromarray(foreground)
            
            # Run detection and segmentation on the foreground
            if mode == OPEN_VOCABULARY_DETECTION:
                _, result = florence_run_inference(
                    model=FLORENCE_MODEL,
                    processor=FLORENCE_PROCESSOR,
                    device=DEVICE,
                    image=image_input,
                    task=TASK_OPEN_VOCABULARY_DETECTION,
                    text=text_input
                )
            elif mode == CAPTION_GROUNDING_MASKS:
                _, result = florence_run_inference(
                    model=FLORENCE_MODEL,
                    processor=FLORENCE_PROCESSOR,
                    device=DEVICE,
                    image=image_input,
                    task=TASK_DETAILED_CAPTION
                )
                caption = result[TASK_DETAILED_CAPTION]
                _, result = florence_run_inference(
                    model=FLORENCE_MODEL,
                    processor=FLORENCE_PROCESSOR,
                    device=DEVICE,
                    image=image_input,
                    task=TASK_CAPTION_TO_PHRASE_GROUNDING,
                    text=caption
                )

            # Convert detections to Supervision Detections
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_input.size
            )
            
            # Perform SAM inference
            detections = perform_sam_inference(SAM_MODEL, image_input, detections)
            
            # Annotate image
            annotated_image = annotate_image(image_input, detections)
            
            # Convert annotated image back to BGR for writing
            output_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
            
            # Optionally, overlay the foreground mask (for visualization)
            # mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # output_frame = cv2.addWeighted(output_frame, 0.8, mask_colored, 0.2, 0)
            
            out.write(output_frame)
            pbar.update(1)

    cap.release()
    out.release()
    print(f"Processing complete. Output saved to {output_video_path}")

# Main function with argument parsing
def main():
    parser = argparse.ArgumentParser(description='Process video with RAFT-based foreground extraction.')
    parser.add_argument('--input_video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to save the output video file')
    parser.add_argument('--mode', type=str, choices=[OPEN_VOCABULARY_DETECTION, CAPTION_GROUNDING_MASKS], default=OPEN_VOCABULARY_DETECTION, help='Processing mode')
    parser.add_argument('--text_input', type=str, help='Text input for detection (required for open vocabulary detection)')

    args = parser.parse_args()

    if args.mode == OPEN_VOCABULARY_DETECTION and args.text_input is None:
        parser.error(f"Text input is required when mode is {OPEN_VOCABULARY_DETECTION}")

    process_video(args.input_video_path, args.output_video_path, args.mode, args.text_input)

if __name__ == "__main__":
    main()
