import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import argparse
from utils.utils import annotate_image
from utils.florence import florence_load_model, florence_run_inference, \
    TASK_DETAILED_CAPTION, \
    TASK_CAPTION_TO_PHRASE_GROUNDING, TASK_OPEN_VOCABULARY_DETECTION
from utils.modes import OPEN_VOCABULARY_DETECTION, CAPTION_GROUNDING_MASKS
from utils.sam import initialize_sam, perform_sam_inference
import supervision as sv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLORENCE_MODEL, FLORENCE_PROCESSOR = florence_load_model(device=DEVICE)
SAM_MODEL = initialize_sam(device=DEVICE)


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
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    with tqdm(total=total_frames, desc="Processing Video", unit="frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_input = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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

            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_input.size
            )
            detections = perform_sam_inference(SAM_MODEL, image_input, detections)
            annotated_image = annotate_image(image_input, detections)
            output_frame = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
            out.write(output_frame)
            pbar.update(1)

    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description='Process video for specific text input.')
    parser.add_argument('--input_video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to save the output video file')
    parser.add_argument('--mode', type=str, choices=[OPEN_VOCABULARY_DETECTION, CAPTION_GROUNDING_MASKS], default=OPEN_VOCABULARY_DETECTION, help='Processing mode')
    parser.add_argument('--text_input', type=str, help='Text input for detection')

    args = parser.parse_args()

    if args.mode == OPEN_VOCABULARY_DETECTION and args.text_input is None:
        parser.error(f"Text input is required when mode is {OPEN_VOCABULARY_DETECTION}")

    process_video(args.input_video_path, args.output_video_path, args.mode, args.text_input)

if __name__ == "__main__":
    main()
