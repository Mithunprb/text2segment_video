import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

# Add RAFT to the system path (adjust the path as necessary)
sys.path.append('/kaggle/input/raft-pytorch')  # Update this path based on your environment

from utils.utils import annotate_image, MASK_ANNOTATOR
from utils.florence import florence_load_model, florence_run_inference, TASK_CAPTION_TO_PHRASE_GROUNDING
from utils.sam import initialize_sam, perform_sam_inference
import supervision as sv

# RAFT Imports
from raft.core.raft import RAFT
from raft.core.utils import flow_viz
from raft.core.utils.utils import InputPadder
from raft.config import RAFTConfig
import matplotlib.pyplot as plt

# DEVICE Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Florence and SAM Models
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
    
    model = RAFT(config)
    weights_path = '/kaggle/input/raft-pytorch/raft-sintel.pth'  # Update the path to RAFT weights
    ckpt = torch.load(weights_path, map_location=device)
    model.to(device)
    model.load_state_dict(ckpt)
    model.eval()
    return model

RAFT_MODEL = initialize_raft(DEVICE)

# Function to Compute Optical Flow and Extract Foreground
def compute_flow_and_foreground(image1, image2, model, device, threshold=2.0):
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
    img1_np = image1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    foreground = cv2.bitwise_and(img1_np, mask_3ch)
    
    return flow_up, mask, foreground

# Visualization Function (Optional)
def viz(img1, img2, flo, mask=None, foreground=None):
    img1 = img1[0].permute(1,2,0).cpu().numpy().astype(int)
    img2 = img2[0].permute(1,2,0).cpu().numpy().astype(int)
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # Map flow to RGB image
    flo_img = flow_viz.flow_to_image(flo)

    fig, axes = plt.subplots(1, 4, figsize=(25, 5))
    axes[0].set_title('Input Image 1')
    axes[0].imshow(img1)
    axes[1].set_title('Input Image 2')
    axes[1].imshow(img2)
    axes[2].set_title('Optical Flow')
    axes[2].imshow(flo_img)
    
    if mask is not None and foreground is not None:
        axes[3].set_title('Foreground Mask')
        axes[3].imshow(mask, cmap='gray')
        # Uncomment below lines if you want to visualize the foreground
        # plt.figure(figsize=(5,5))
        # plt.title('Extracted Foreground')
        # plt.imshow(foreground)
    
    plt.show()

def process_video(input_video_path, output_video_path, mask_video_path, text_input=None):
    if not os.path.exists(input_video_path):
        print("Error: The file does not exist.")
        return
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_width * frame_height
    large_detection_threshold = frame_area * 0.75  

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
    mask_out = cv2.VideoWriter(mask_video_path, fourcc, 20.0, (frame_width, frame_height))

    # Initialize RAFT by reading the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        return
    prev_frame_rgb = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    prev_image = torch.from_numpy(prev_frame_rgb).permute(2, 0, 1).float().to(DEVICE)
    prev_image = prev_image[None]  # Add batch dimension

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, desc="Processing Video", unit="frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_image = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().to(DEVICE)
            current_image = current_image[None]  # Add batch dimension

            # Compute optical flow and extract foreground
            flow_up, mask, foreground = compute_flow_and_foreground(prev_image, current_image, RAFT_MODEL, DEVICE, threshold=2.0)
            
            # Determine if there's significant motion based on the mask
            significant_motion = cv2.countNonZero(mask) > (frame_area // 90)
            
            if significant_motion:
                image_input = Image.fromarray(frame_rgb)
                _, response = florence_run_inference(
                    model=FLORENCE_MODEL,
                    processor=FLORENCE_PROCESSOR,
                    device=DEVICE,
                    image=image_input,
                    task=TASK_CAPTION_TO_PHRASE_GROUNDING,
                    text=text_input
                )

                bbox_data = response.get(TASK_CAPTION_TO_PHRASE_GROUNDING, {}).get('bboxes', [])
                for bbox in bbox_data:
                    # Calculate the absolute area of the bounding box
                    bbox_area = bbox[2] * bbox[3]  # width * height from the response
                    if bbox_area > large_detection_threshold:
                        continue  # Skip SAM processing if the detected area is too large
                    else:
                        detections = sv.Detections.from_lmm(
                            lmm=sv.LMM.FLORENCE_2,
                            result=response,
                            resolution_wh=image_input.size
                        )
                        detections, score = perform_sam_inference(SAM_MODEL, image_input, detections)
                        annotated_image = annotate_image(image_input, detections)
                        mask_image = MASK_ANNOTATOR.annotate(image_input.copy(), detections)

                        output_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
                        mask_frame = cv2.cvtColor(np.array(mask_image), cv2.COLOR_RGB2BGR)
                        out.write(output_frame)
                        mask_out.write(mask_frame)
            else:
                out.write(frame)
                mask_out.write(frame)
            
            # Update previous frame
            prev_image = current_image

            pbar.update(1)

    cap.release()
    out.release()
    mask_out.release()

def main():
    parser = argparse.ArgumentParser(description='Process video for specific text input using RAFT optical flow.')
    parser.add_argument('--input_video_path', type=str, default='vid_src/6_new.mp4', help='Path to the input video file')
    parser.add_argument('--output_video_path', type=str, default='output.mp4', help='Path to save the output video file')
    parser.add_argument('--mask_video_path', type=str, default='mask_output.mp4', help='Path to save the mask video file')
    parser.add_argument('--text_input', type=str, default='person carrying a weapon', help='Text input for processing')

    args = parser.parse_args()

    process_video(args.input_video_path, args.output_video_path, args.mask_video_path, args.text_input)

if __name__ == "__main__":
    main()
