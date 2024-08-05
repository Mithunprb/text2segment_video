from typing import Tuple, Optional
import torch
from PIL import Image
from utils.utils import annotate_image, show_image
import argparse
from utils.florence import florence_load_model, florence_run_inference, \
    TASK_DETAILED_CAPTION, \
    TASK_CAPTION_TO_PHRASE_GROUNDING, TASK_OPEN_VOCABULARY_DETECTION
from utils.sam import initialize_sam, perform_sam_inference
import supervision as sv

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FLORENCE_MODEL, FLORENCE_PROCESSOR = florence_load_model(device=DEVICE)
SAM_MODEL = initialize_sam(device=DEVICE)


OPEN_VOCABULARY_DETECTION = "open vocabulary detection + masks"
CAPTION_GROUNDING_MASKS = "caption + grounding + masks"

INFERENCE_MODES = [
    OPEN_VOCABULARY_DETECTION,
    CAPTION_GROUNDING_MASKS
]


def process(mode, image_path, text_input=None) -> Tuple[Optional[Image.Image], Optional[str]]:
    image_input = Image.open(image_path)

    if mode == OPEN_VOCABULARY_DETECTION:
        if not text_input:
            return None, "Text input required for this mode."

        _, result = florence_run_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=TASK_OPEN_VOCABULARY_DETECTION,
            text=text_input
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        detections = perform_sam_inference(SAM_MODEL, image_input, detections)
        annotated_image = annotate_image(image_input, detections)
        show_image(annotated_image)
        return annotated_image, None

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
        show_image(annotated_image)
        return annotated_image, caption

def main():
    parser = argparse.ArgumentParser(description='Process images with different modes.')
    parser.add_argument('--mode', type=int, required=True, help='Select a mode by index (1 for DETECTION_MODE, 2 for CLASSIFICATION_MODE, 3 for OPEN_VOCABULARY_DETECTION)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    parser.add_argument('--text_input', type=str, help='Text input for OPEN_VOCABULARY_DETECTION mode')

    args = parser.parse_args()

    mode_index = args.mode - 1
    if mode_index < 0 or mode_index >= len(INFERENCE_MODES):
        print("Invalid mode selected. Please select a valid mode index.")
        return

    mode = INFERENCE_MODES[mode_index]
    image_path = args.image_path
    text_input = args.text_input

    if mode == OPEN_VOCABULARY_DETECTION and text_input is None:
        print("Text input is required for OPEN_VOCABULARY_DETECTION mode.")
        return

    image_output, text_output = process(mode, image_path, text_input)

    if text_output:
        print("Caption output:", text_output)

if __name__ == "__main__":
    main()
