import sys, time
import os
import cv2
from modules.batch_processor import BatchProcessor
from dataclasses import dataclass
from typing import Dict

from modules.inpainting.schema import Config
from modules.maga_settings import PSettings

@dataclass
class RenderSettings:
    font_family: str = 'Verdana'  # '/Users/mac/ocode/RiseInRose/comic-translate/fonts/文津宋体 第0平面_mianfeiziti.com.ttf' #
    color: str = '#111'
    line_spacing: float = 1.0
    outline: bool = True
    outline_color: str = '#fff'
    outline_width: float = 1.0
    bold: bool = False
    italic: bool = False
    underline: bool = False
    alignment: str = 'center'
    direction: str = 'auto'
    max_font_size: int = 40
    min_font_size: int = 16
    upper_case: bool = False

@dataclass
class Settings:
    gpu_enabled: bool = True
    lang_mapping: Dict[str, str] = None
    inpainter_key: str = 'LaMa'  # LaMa、MI-GAN 、AOT
    inpaint_config: Config = None
    export_inpainted_image: bool = False
    export_raw_text: bool = False
    export_translated_text: bool = False
    render_settings: RenderSettings = None
    settings_page: PSettings = None

    def __init__(self, source_language, target_language):
        self.settings_page = PSettings()

        if self.lang_mapping is None:
            self.lang_mapping = {
                self.settings_page.tr("English"): "English",
                self.settings_page.tr("Korean"): "Korean",
                self.settings_page.tr("Japanese"): "Japanese",
                self.settings_page.tr("French"): "French",
                self.settings_page.tr("Simplified Chinese"): "Simplified Chinese",
                self.settings_page.tr("Traditional Chinese"): "Traditional Chinese",
                self.settings_page.tr("Chinese"): "Chinese",
                self.settings_page.tr("Russian"): "Russian",
                self.settings_page.tr("German"): "German",
                self.settings_page.tr("Dutch"): "Dutch",
                self.settings_page.tr("Spanish"): "Spanish",
                self.settings_page.tr("Italian"): "Italian",
                self.settings_page.tr("Turkish"): "Turkish",
                self.settings_page.tr("Polish"): "Polish",
                self.settings_page.tr("Portuguese"): "Portuguese",
                self.settings_page.tr("Brazilian Portuguese"): "Brazilian Portuguese",
                self.settings_page.tr("Thai"): "Thai",
                self.settings_page.tr("Vietnamese"): "Vietnamese",
                self.settings_page.tr("Indonesian"): "Indonesian",
                self.settings_page.tr("Hungarian"): "Hungarian",
                self.settings_page.tr("Finnish"): "Finnish",
                self.settings_page.tr("Arabic"): "Arabic",
                self.settings_page.tr("Hindi"): "Hindi",
            }
        if self.inpaint_config is None:
            self.inpaint_config = Config()
        if self.render_settings is None:
            self.render_settings = RenderSettings()

        from pathlib import Path
        src_folder = Path(__file__).parent
        if 'Traditional Chinese' == target_language:
            self.render_settings.font_family = os.path.join(src_folder, 'fonts/ToneOZ-Tsuipita-TC.ttf')
        elif 'Chinese' in target_language:
            self.render_settings.font_family = os.path.join(src_folder, 'fonts/ToneOZ-Tsuipita-TC.ttf')
        elif target_language == 'Japanese':
            self.render_settings.font_family = os.path.join(src_folder, 'fonts/KleeOne-SemiBold.ttf')
        elif target_language == 'Korean':
            self.render_settings.font_family = os.path.join(src_folder, 'fonts/NotoSerifCJKkr-SemiBold.otf')
        elif target_language in {'English', 'French', 'German', 'Spanish', 'Italian', 'Dutch', 'Portuguese',
                                 'Brazilian Portuguese'}:
            self.render_settings.font_family = os.path.join(src_folder, 'fonts/anime_ace_3.ttf')
        else:
            self.render_settings.font_family = os.path.join(src_folder, 'fonts/Arial-Unicode-Regular.ttf')
        print('font file:', self.render_settings.font_family)

        # self.settings_page.tools.translator = 'Deepseek'
        # self.settings_page.credentials.credentials['Deepseek_api_key'] = 'sk-91043ea797ae460680043f6964239dc1'
        print('comic_open_api_key', os.getenv('comic_open_api_key'))
        self.settings_page.tools.translator = 'Custom'
        self.settings_page.credentials.credentials['Custom_api_key'] = 'sk-eb7bza776ztE5ieH3e0aA56c7aB24bA38dFf734cB50f4d69'#'sk-jrKOlAUbyydnF6ve4e751a56Cb564eAd90A100671942Ae7c'#'sk-6rPJZBY5dUqPvwEaCf4353CaC9Ae465091Ac2a79510187Dc'
        self.settings_page.credentials.credentials['Custom_api_url'] = 'https://api.mixrai.com/v1'
        self.settings_page.credentials.credentials['Custom_model'] = 'gpt-4o-mini'
        self.settings_page.credentials.credentials['Custom_openai_api_key'] = os.getenv('comic_open_api_key')
        self.settings_page.credentials.credentials['Custom_openai_api_url'] = 'https://api.openai.com/v1'
        self.settings_page.credentials.credentials['Open AI GPT_api_key'] = os.getenv('comic_open_api_key')

        self.settings_page.llm.extra_context = f'''You are an expert translator who translates {source_language} to {target_language}. You pay attention to style, formality, idioms, slang etc and try to convey it in the way a {target_language} speaker would understand.\n        BE MORE NATURAL. NEVER USE 당신, 그녀, 그 or its {source_language} equivalen        Specifically, you will be translating text OCR'd from a comic. The OCR is not perfect and as such you may receive text with typos or other mistakes.\n        To aid you and provide context, You may be given the image of the page and/or extra context about the comic. You will be given a json string of the detected text blocks and the text to translate. Return the json string with the texts translated. DO NOT translate the keys of the json. For each block:\n        - If it's already in {target_language}, OUTPUT IT AS IT IS instead\n        - DO NOT give explanations\n        Do Your Best! I'm really counting on you.'''

def main():
    if len(sys.argv) < 2:
        print("请提供图片文件路径作为参数")
        return
    
    image_files = []
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            image_files.append(arg)
    
    if not image_files:
        print("未找到有效的图片文件")
        return

    source_language = 'Japanese'
    target_language = 'Chinese' # 'English' #
    # 构造基本设置
    settings = Settings(source_language, target_language)
    
    # 构造图片状态信息
    image_states = {}
    for image_path in image_files:
        image_states[image_path] = {
            'source_lang': source_language,  # 默认源语言
            'target_lang': target_language,  # 目标语言
            'viewer_state': {}
        }
    
    # 初始化处理器并处理图片
    processor = BatchProcessor()
    processor.process_images(
        image_files=image_files,
        image_states=image_states,
        settings=settings
    )


def run(input_path, output_path, target_language, source_language='Japanese', logger=None):
    image_files = []

    names = os.listdir(input_path)
    for name in names:
        full_image_name = os.path.join(input_path, name)
        image_files.append(full_image_name)

    # 构造基本设置
    settings = Settings(source_language, target_language)

    # 构造图片状态信息
    image_states = {}
    for image_path in image_files:
        image_states[image_path] = {
            'source_lang': source_language,  # 默认源语言
            'target_lang': target_language,  # 目标语言
            'viewer_state': {}
        }

    print('image_states', image_states)
    t1 = time.time()
    # 初始化处理器并处理图片
    processor = BatchProcessor()
    flag, msg = processor.process_images(
        image_files=image_files,
        image_states=image_states,
        settings=settings,
        output_path=output_path,
        logger=logger
    )
    print('----full process time----%s' % (time.time() - t1))
    return flag, msg

if __name__ == '__main__':
    # main()

    if len(sys.argv) < 2:
        import logging
        logger = logging.getLogger('test')
        input_path = '/Users/mac/Documents/manga/test2/input'
        output_path = '/Users/mac/Documents/manga/test2/output'
        source_language = 'English'
        target_language = 'Simplified Chinese'
        run(input_path, output_path, target_language, source_language, logger=logger)
    else:
        image_path = sys.argv[1]
        input_image = cv2.imread(image_path)

        # source_language = 'Traditional Chinese'
        source_language = 'Japanese'
        target_language = 'Hindi'  # 'Russian' # 'Korean' # 'Simplified Chinese' # 'Chinese' # 'English' #'Traditional Chinese' # 'Japanese'
        settings = Settings(source_language, target_language)
        processor = BatchProcessor()
        # flag, output_image = processor.process_one_image(settings, input_image, 'Japanese', target_language)
        t1 = time.time()
        # flag, covert_image = processor.run_render_block(settings, input_image, source_language, target_language)
        # output_image = covert_image

        # cv2.imwrite('covert_image.png', covert_image)
        # covert_input_image = cv2.imread('covert_image.png')
        # import logging
        # logger = logging.getLogger('test')
        # flag, output_image = processor.process_one_image(settings, covert_input_image, source_language, target_language, logger=logger)
        # for y1, y2, sy1, sy2 in clip_arrs:
        #     input_image[sy1+3:sy2-3,:,:] = output_image[y1+3:y2-3,:,:]
        # output_image = input_image

        import logging
        logger = logging.getLogger('test')
        flag, output_image = processor.process_one_image(settings, input_image, source_language, target_language, logger=logger)

        if flag:
            cv2.imwrite('output.png', output_image)

        print('------full-time------%s' % (time.time() - t1))




