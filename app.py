import streamlit as st
import os
import numpy as np
import cv2
from datetime import datetime
from PIL import Image
import shutil
from main import main
from supplementary.our_args import args

# Setting the page configuration with an icon
image_directory = "supplementary/vs_clip.png"
image = Image.open(image_directory)
PAGE_CONFIG = {
    "page_title": "Video Synopsis",
    "page_icon": image,
    "layout": "wide",
    "initial_sidebar_state": "auto"
}
st.set_page_config(**PAGE_CONFIG)

def setup_environment(args):
    # Path settings and directory cleanup
    output_path = args["output"]
    final = args["masks"]
    synopsis_frames = args["synopsis_frames"]
    for path in [output_path, synopsis_frames, final]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    os.chdir(output_path)

    # Video capture and background preparation
    cap = cv2.VideoCapture(args['video'])
    cap1 = cv2.VideoCapture(args['video'])
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    frame_width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_length = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    bgimg = prepare_background_image(cap1, fps)
    print(f'[original video] frame_width: {frame_width}, frame_height: {frame_height} \u2705')
    print(f'[original video] Total frames: {video_length} \u2705')
    print(f'[original video] FPS: {fps} \u2705')
    return cap, video_length, bgimg, fps

def prepare_background_image(cap, fps):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rand_indices = np.random.choice(total_frames, size=fps, replace=False)
    Sframes = [
        cap.read()[1] 
        for _ in rand_indices 
        if cap.set(cv2.CAP_PROP_POS_FRAMES, _)
    ]

    if Sframes:
        median_frame = np.median(np.array(Sframes), axis=0).astype(np.uint8)

        # Ensure the parent directory of bg_path exists
        bg_path = args['bg_path']
        os.makedirs(os.path.dirname(bg_path), exist_ok=True)

        cv2.imwrite(bg_path, median_frame)
        bgimg = cv2.cvtColor(np.asarray(Image.open(bg_path)), cv2.COLOR_RGB2BGR)
        return bgimg
    else:
        raise ValueError("[Error]: Unable to calculate median frame. No valid frames were sampled.")


def run_main(args):
    cap, video_length, bgimg, fps = setup_environment(args)
    final = args['masks']
    temp_video_name = os.path.abspath(f"../{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}_temp.mp4")
    final_video_name = temp_video_name.replace('_temp', '')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(temp_video_name, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    main(args, cap, video, video_length, final, bgimg, args['energy_opt'], args['epochs'], final_video_name)
    
    cap.release()
    video.release()

    if os.path.exists(temp_video_name):
        command = f'ffmpeg -loglevel error -i "{temp_video_name}" -vcodec libx264 -crf 23 -preset fast "{final_video_name}"'
        os.system(command)
        if os.path.exists(final_video_name):
            os.remove(temp_video_name)
            st.video(final_video_name)
        else:
            st.error('Failed to process video correctly with FFmpeg. \u274C')
    else:
        st.error(f'[Info] Video file not found at {temp_video_name} \u274C')

def handle_file_upload(uploaded_file):
    if uploaded_file:
        filename = os.path.abspath(uploaded_file.name)
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return filename

with st.form("input_form"):
    st.title("Video Synopsis Configuration")

    uploaded_file = st.file_uploader("Upload Video:", type=['mp4', 'avi'])
    video_path = handle_file_upload(uploaded_file) if uploaded_file else None
    if video_path:
        _, left_col, _ = st.columns([1, 8, 1])
        with left_col:
            st.subheader("Original Video")
            st.video(video_path, format="video/mp4", start_time=0)

    
    col1, col2, col3 = st.columns(3)
    with col1:
        buff_size = st.number_input("Buffer Size:", value=32, format="%d")
        input_model = st.text_input("Input Model:", value='Unet_2020-07-20')
    with col2:
        ext = st.text_input("Extract Object Extension:", value='.png')
        dvalue = st.number_input("Compression Value:", value=9, min_value=0, max_value=9, format="%d")
    with col3:
        energy_opt = st.checkbox("Optimize Energy:", value=True)
        epochs = st.number_input("Epochs:", value=1000, min_value=1, format="%d")

    submitted = st.form_submit_button("Run")

if submitted and video_path:
    # Update the args dynamically without redefining
    args.update({
        'video': video_path,
        'buff_size': buff_size,
        'input_model': input_model,
        'ext': ext,
        'dvalue': dvalue,
        'energy_opt': energy_opt,
        'epochs': epochs
    })

    output_video = run_main(args)
    if output_video:
        _, right_col, _ = st.columns([1, 8, 1])
        with right_col:
            st.subheader("Video Synopsis")
            st.video(output_video)
else:
    if submitted:
        st.error("Please upload a video file to proceed. \u274C")
