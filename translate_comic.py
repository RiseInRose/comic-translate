import os
import sys
import cv2
import argparse
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QColor

from controller import ComicTranslate
from modules.detection import TextBlockDetector
from modules.ocr.processor import OCRProcessor
from modules.translator.processor import Translator
from modules.utils.textblock import sort_blk_list
from modules.utils.pipeline_utils import inpaint_map, get_config, generate_mask, get_language_code
from modules.rendering.render import get_best_render_area, pyside_word_wrap
from modules.utils.translator_utils import format_translations
from app.ui.canvas.save_renderer import ImageSaveRenderer
from app.ui.canvas.text_item import OutlineInfo, OutlineType

def parse_args():
    parser = argparse.ArgumentParser(description='Translate comic images from command line')
    parser.add_argument('input_path', help='Path to input image or directory')
    parser.add_argument('--output_dir', help='Output directory path', default='output')
    parser.add_argument('--src_lang', help='Source language', default='Korean')
    parser.add_argument('--tgt_lang', help='Target language', default='English')
    parser.add_argument('--gpu', help='Use GPU', action='store_true')
    return parser.parse_args()

def process_image(image_path, output_dir, src_lang, tgt_lang, use_gpu):
    # Initialize QApplication (required for font rendering)
    app = QApplication(sys.argv)
    
    # Initialize main controller (needed for settings)
    controller = ComicTranslate()
    settings_page = controller.settings_page
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Initialize detectors
    device = 0 if use_gpu else 'cpu'
    detector = TextBlockDetector(
        'models/detection/comic-speech-bubble-detector.pt',
        'models/detection/comic-text-segmenter.pt',
        'models/detection/manga-text-detector.pt',
        device
    )
    
    # Detect text blocks
    print("Detecting text blocks...")
    blk_list = detector.detect(image)
    if not blk_list:
        print("No text blocks detected")
        return
    
    # OCR
    print("Performing OCR...")
    ocr = OCRProcessor()
    ocr.initialize(controller, src_lang)
    ocr.process(image, blk_list)
    
    # Sort blocks
    rtl = True if src_lang == 'Japanese' else False
    blk_list = sort_blk_list(blk_list, rtl)
    
    # Inpainting
    print("Inpainting...")
    inpainter_key = settings_page.get_tool_selection('inpainter')
    InpainterClass = inpaint_map[inpainter_key]
    inpainter = InpainterClass(device)
    
    config = get_config(settings_page)
    mask = generate_mask(image, blk_list)
    inpainted = inpainter(image, mask, config)
    inpainted = cv2.convertScaleAbs(inpainted)
    
    # Translation
    print("Translating text...")
    translator = Translator(controller, src_lang, tgt_lang)
    translator.translate(blk_list, image, "")
    
    # Format translations
    trg_lng_cd = get_language_code(tgt_lang)
    format_translations(blk_list, trg_lng_cd)
    
    # Get rendering area
    get_best_render_area(blk_list, image, inpainted)
    
    # Setup rendering parameters
    render_settings = {
        'font_family': 'Arial',
        'font_size': 20,
        'color': '#000000',
        'line_spacing': 1.0,
        'outline': True,
        'outline_color': '#FFFFFF',
        'outline_width': 1.0,
        'bold': False,
        'italic': False,
        'underline': False,
        'alignment': Qt.AlignmentFlag.AlignCenter,
        'direction': Qt.LayoutDirection.LeftToRight
    }
    
    # Render text
    print("Rendering translated text...")
    text_items_state = []
    for blk in blk_list:
        x1, y1, width, height = blk.xywh
        translation = blk.translation
        if not translation or len(translation) == 1:
            continue
            
        translation, font_size = pyside_word_wrap(
            translation, 
            render_settings['font_family'],
            width, height,
            render_settings['line_spacing'],
            render_settings['outline_width'],
            render_settings['bold'],
            render_settings['italic'],
            render_settings['underline'],
            render_settings['alignment'],
            render_settings['direction'],
            40,  # max font size
            12   # min font size
        )
        
        if any(lang in trg_lng_cd.lower() for lang in ['zh', 'ja', 'th']):
            translation = translation.replace(' ', '')
            
        text_items_state.append({
            'text': translation,
            'font_family': render_settings['font_family'],
            'font_size': font_size,
            'text_color': QColor(render_settings['color']),
            'alignment': render_settings['alignment'],
            'line_spacing': render_settings['line_spacing'],
            'outline_color': QColor(render_settings['outline_color']),
            'outline_width': render_settings['outline_width'],
            'bold': render_settings['bold'],
            'italic': render_settings['italic'],
            'underline': render_settings['underline'],
            'position': (x1, y1),
            'rotation': blk.angle,
            'scale': 1.0,
            'transform_origin': blk.tr_origin_point,
            'width': width,
            'direction': render_settings['direction'],
            'selection_outlines': [OutlineInfo(0, len(translation), 
                                             QColor(render_settings['outline_color']),
                                             render_settings['outline_width'],
                                             OutlineType.Full_Document)] if render_settings['outline'] else []
        })
    
    # Save translated image
    print("Saving translated image...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_translated.png")
    
    renderer = ImageSaveRenderer(inpainted)
    renderer.add_state_to_image({'text_items_state': text_items_state})
    renderer.save_image(output_path)
    
    print(f"Translation completed. Output saved to: {output_path}")

def main():
    args = parse_args()
    
    if os.path.isfile(args.input_path):
        # Process single image
        process_image(args.input_path, args.output_dir, args.src_lang, args.tgt_lang, args.gpu)
    elif os.path.isdir(args.input_path):
        # Process all images in directory
        for filename in os.listdir(args.input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_path = os.path.join(args.input_path, filename)
                print(f"\nProcessing: {filename}")
                process_image(image_path, args.output_dir, args.src_lang, args.tgt_lang, args.gpu)
    else:
        print("Invalid input path")

if __name__ == "__main__":
    main() 