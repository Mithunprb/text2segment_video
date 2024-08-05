import supervision as sv
import matplotlib.pyplot as plt

MASK_ANNOTATOR = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
BOX_ANNOTATOR = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#FFFFFF"),
    border_radius=5
)

def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image

def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()