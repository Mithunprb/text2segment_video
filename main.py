import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import argparse

from utils.utils import annotate_image, MASK_ANNOTATOR
from utils.florence import florence_load_model, florence_run_inference,  \
    TASK_CAPTION_TO_PHRASE_GROUNDING

from utils.sam import initialize_sam, perform_sam_inference
import supervision as sv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLORENCE_MODEL, FLORENCE_PROCESSOR = florence_load_model(device=DEVICE)
SAM_MODEL = initialize_sam(device=DEVICE)

def process_video(input_video_path, output_video_path, mask_video_path, text_input=None):
    if not os.path.exists(input_video_path):
        print("Error: The file does not exist.")
        return
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_area = frame_width * frame_height
    large_detection_threshold = frame_area * 0.75  

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
    mask_out = cv2.VideoWriter(mask_video_path, fourcc, 20.0, (frame_width, frame_height))

    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing Video", unit="frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            fg_mask = background_subtractor.apply(frame)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            significant_motion = any(cv2.contourArea(contour) > frame_area // 90 for contour in contours)

            if significant_motion:
                image_input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
                
            pbar.update(1)

    cap.release()
    out.release()
    mask_out.release()

def main():
    parser = argparse.ArgumentParser(description='Process video for specific text input.')
    parser.add_argument('--input_video_path', type=str, default='vid_src/6_new.mp4', help='Path to the input video file')
    parser.add_argument('--output_video_path', type=str, default='output.mp4', help='Path to save the output video file')
    parser.add_argument('--mask_video_path', type=str, default='mask_output.mp4', help='Path to save the mask video file')
    parser.add_argument('--text_input', type=str, default='person carrying a weapon', help='Text input for processing')

    args = parser.parse_args()

    process_video(args.input_video_path, args.output_video_path, args.mask_video_path, args.text_input)

if __name__ == "__main__":
    main()
