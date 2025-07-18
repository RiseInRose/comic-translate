import cv2
import numpy as np
from typing import Tuple, List

from PIL import Image, ImageFont, ImageDraw
from .hyphen_textwrap import wrap as hyphen_wrap
from .render import pil_word_wrap
from ..utils.textblock import TextBlock
from ..detection import make_bubble_mask, bubble_interior_bounds
from ..utils.textblock import adjust_blks_size

class TextRenderingSettings:
    alignment: str
    font_family: str
    min_font_size: int
    max_font_size: int
    color: str
    upper_case: bool
    outline: bool
    outline_color: str
    outline_width: int
    bold: bool
    italic: bool
    underline: bool
    line_spacing: int
    direction: str

def cv2_to_pil(cv2_image: np.ndarray):
    # Convert color channels from BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def pil_to_cv2(pil_image: Image):
    # Convert the PIL image to a numpy array
    numpy_image = np.array(pil_image)
    
    # PIL images are in RGB by default, OpenCV uses BGR, so convert the color space
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
    return cv2_image

def pyside_word_wrap(text: str, font_pth: str, roi_width: int, roi_height: int,
                    line_spacing: int, outline_width: int, bold: bool, italic: bool, underline: bool,
                    alignment: str, direction: str, init_font_size: int, min_font_size: int = 10, break_long_words=False) -> Tuple[str, int, float, float]:
    """Break long text to multiple lines, and reduce point size
    until all text fits within a bounding box."""

    def prepare_font(font_size):
        font = ImageFont.truetype(font_pth, font_size)
        # PIL doesn't support font styles directly, would need to use specific font files
        return font

    def eval_metrics(txt: str, font_sz: float) -> Tuple[float, float]:
        """Quick helper function to calculate width/height of text using PIL."""
        # Create a temporary image for text measurement
        temp_img = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(temp_img)
        font = prepare_font(font_sz)

        # Get text bounds
        left, top, right, bottom = draw.multiline_textbbox((0, 0), txt, font=font, align=alignment, spacing=line_spacing)
        width = right - left
        height = bottom - top

        # Add outline width to the size
        if outline_width > 0:
            width += 2 * outline_width
            height += 2 * outline_width

        return width, height

    text = text.replace('．', '.').replace('. ', '.').replace('...', ' ... ').strip()

    mutable_message = text
    font_size = init_font_size
    block_area = roi_width * roi_height
    width, height = eval_metrics(mutable_message, font_size)
    word_area = width * height
    area_ratio = word_area * 1.0 / block_area
    if area_ratio > 1:
        font_size = int(font_size / area_ratio)
        if font_size < min_font_size:
            font_size = min_font_size

    mutable_message = '\n'.join(
        hyphen_wrap(text, 1, break_on_hyphens=True, break_long_words=break_long_words, hyphenate_broken_words=True))
    wrapped_width, _ = eval_metrics(mutable_message, font_size)
    area_ratio = wrapped_width * 1.0 / roi_width
    if area_ratio > 1:
        font_size = int(font_size / area_ratio)
        if font_size < min_font_size:
            font_size = min_font_size

    while font_size > min_font_size:
        width, height = eval_metrics(mutable_message, font_size)
        if height > roi_height:
            font_size -= 1  # Reduce pointsize
            mutable_message = text  # Restore original text
        elif width > roi_width:
            columns = len(mutable_message)
            while columns > 0:
                if columns > 4:
                    columns //= 2
                else:
                    columns -= 1
                if columns == 0:
                    break
                mutable_message = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=True, break_long_words=break_long_words, hyphenate_broken_words=True))
                wrapped_width, _ = eval_metrics(mutable_message, font_size)
                if wrapped_width <= roi_width:
                    break
            if columns < 1:
                font_size -= 1  # Reduce pointsize
                mutable_message = text  # Restore original text
        else:
            break

    if font_size <= min_font_size:
        font_size = min_font_size
        mutable_message = text

        # Wrap text to fit within as much as possible
        # Minimize cost function: (width - roi_width)^2 + (height - roi_height)^2
        min_cost = 1e9
        min_text = text
        for columns in range(1, len(text)):
            wrapped_text = '\n'.join(hyphen_wrap(text, columns, break_on_hyphens=True, break_long_words=break_long_words, hyphenate_broken_words=True))
            wrapped_width, wrapped_height = eval_metrics(wrapped_text, font_size)
            cost = (wrapped_width - roi_width)**2 + (wrapped_height - roi_height)**2
            if cost < min_cost:
                min_cost = cost
                min_text = wrapped_text

        mutable_message = min_text

    width, height = eval_metrics(mutable_message, font_size)
    return mutable_message, font_size, width, height

def draw_text(image: np.ndarray, blk_list: List[TextBlock], font_pth: str, colour: str = "#000", init_font_size: int = 40, min_font_size=10, outline: bool = True):
    image = cv2_to_pil(image)
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font_pth, size=init_font_size)

    for blk in blk_list:
        x1, y1, width, height = blk.xywh
        tbbox_top_left = (x1, y1)

        translation = blk.translation
        if not translation or len(translation) == 1:
            continue

        if blk.min_font_size > 0:
            min_font_size = blk.min_font_size
        if blk.max_font_size > 0:
            init_font_size = blk.max_font_size
        if blk.font_color:
            colour = blk.font_color

        translation, font_size = pil_word_wrap(image, tbbox_top_left, font_pth, translation, width, height,
                                               align=blk.alignment, spacing=blk.line_spacing, init_font_size=init_font_size, min_font_size=min_font_size)
        font = font.font_variant(size=font_size)

        # Font Detection Workaround. Draws white color offset around text
        if outline:
            offsets = [(dx, dy) for dx in (-2, -1, 0, 1, 2) for dy in (-2, -1, 0, 1, 2) if dx != 0 or dy != 0]
            for dx, dy in offsets:
                draw.multiline_text((tbbox_top_left[0] + dx, tbbox_top_left[1] + dy), translation, font=font, fill="#FFF", align=blk.alignment, spacing=1)
        draw.multiline_text(tbbox_top_left, translation, colour, font, align=blk.alignment, spacing=1)
        
    image = pil_to_cv2(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_best_render_area(blk_list: List[TextBlock], img, inpainted_img):
    # Using Speech Bubble detection to find best Text Render Area
    for blk in blk_list:
        if blk.text_class == 'text_bubble':
            bx1, by1, bx2, by2 = blk.bubble_xyxy
            bubble_clean_frame = inpainted_img[by1:by2, bx1:bx2]
            bubble_mask = make_bubble_mask(bubble_clean_frame)
            text_draw_bounds = bubble_interior_bounds(bubble_mask)

            bdx1, bdy1, bdx2, bdy2 = text_draw_bounds

            bdx1 += bx1
            bdy1 += by1

            bdx2 += bx1
            bdy2 += by1

            if blk.source_lang == 'ja':
                blk.xyxy[:] = [bdx1, bdy1, bdx2, bdy2]
                adjust_blks_size(blk_list, img, -5, -5)
            else:
                tx1, ty1, tx2, ty2  = blk.xyxy

                nx1 = max(bdx1, tx1)
                nx2 = min(bdx2, tx2)
                
                blk.xyxy[:] = [nx1, ty1, nx2, ty2]

    return blk_list

