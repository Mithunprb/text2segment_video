# Simple Video Summarization using Text-to-Segment Anything (Florence2 + SAM2)

This project provides a video processing tool that utilizes advanced AI models, specifically Florence2 and SAM2, to detect and segment specific objects or activities in a video based on textual descriptions. The system identifies significant motion in video frames and then performs deep learning inference to locate objects or actions described by the user's textual input.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PcK_6anMRYnRcmOw5TkwFUT8lrXYoHUW?usp=sharing)

[<img src="https://kaggle.com/static/images/site-logo.png" height="50" style="margin-bottom:-15px" />](https://www.kaggle.com/code/mithunparab/simple-video-summarization-using-text-to-segment-a)

## Installation

Before running the script, ensure that all dependencies are installed. You can install the necessary packages using the following command:

```bash
pip install -r requirements.txt
```

For checkpoints:

```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

### Requirements

- Python 3.7+
- OpenCV
- PIL
- Torch
- tqdm
and

```
pip install -q einops spaces timm transformers samv2 gradio supervision opencv-python
```

## Usage

The script can be executed from the command line with arguments to specify the paths of the input video, output video, and mask video, along with the text input for processing.

```bash
python main.py --input_video_path <path_to_input_video> --output_video_path <path_to_output_video> --mask_video_path <path_to_mask_video> --text_input "your text here"
```

### Parameters

- `--input_video_path`: Path to the source video file.
- `--output_video_path`: Path to save the processed video file.
- `--mask_video_path`: Path to save the mask video file that highlights detected objects.
- `--text_input`: Textual description of the object or activity to detect and segment in the video.

The `video_flow.py` script processes videos using RAFT-based optical flow for foreground extraction. It can be executed from the command line with arguments to specify the paths of the input and output video files, the processing mode, and optional text input for detection.

### Arguments

- `--input_video_path`  
  **Required.** Path to the input video file.  
  Example: `--input_video_path ./input_video.mp4`

- `--output_video_path`  
  **Required.** Path to save the output video file.  
  Example: `--output_video_path ./output_video.mp4`

- `--mode`  
  **Required.** Processing mode to specify the type of operation:
  - `OPEN_VOCABULARY_DETECTION`: Detects objects based on text input.
  - `CAPTION_GROUNDING_MASKS`: Processes masks for grounding captions.  
  Example: `--mode OPEN_VOCABULARY_DETECTION`

- `--text_input`  
  **Optional (Required for `OPEN_VOCABULARY_DETECTION` mode).** Text input for detecting specific objects in the video.  
  Example: `--text_input "person walking"`

### Notes

1. When using `OPEN_VOCABULARY_DETECTION` mode, the `--text_input` argument is required.  
2. Ensure RAFT is properly set up in your environment for accurate foreground extraction.

### Full Example

```bash
python video_flow.py --input_video_path <path_to_input_video> --output_video_path <path_to_output_video> --mask_video_path <path_to_mask_video> --text_input "your text here"
```

## Features

- **Motion Detection**: Detect significant motions in the video to focus processing on relevant segments.
- **Object and Action Detection**: Utilize state-of-the-art models (Florence2 and SAM2) to detect and segment objects or actions specified by the user.
- **Video and Mask Output**: Generate an annotated video and a corresponding mask video showing the detected segments.

## To Do

- [ ] **WebUI**
- [ ] **Robust Video Synopsis**

## Related work

- <https://github.com/facebookresearch/segment-anything-2/tree/main>
- <https://huggingface.co/spaces/SkalskiP/florence-sam>
- <https://huggingface.co/collections/microsoft/florence-6669f44df0d87d9c3bfb76de>
