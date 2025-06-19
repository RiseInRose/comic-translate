import os, time
import cv2
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Callable

from modules.detection import TextBlockDetector
from modules.ocr.maga_processor import OCRProcessor
from modules.save_renderer import ImageSaveRenderer
from modules.translator.maga_processor import Translator
from modules.utils.textblock import TextBlock, sort_blk_list
from modules.utils.maga_utils import inpaint_map, get_config
from modules.rendering.maga_render import get_best_render_area, pyside_word_wrap
from modules.utils.maga_utils import generate_mask, get_language_code, is_directory_empty
from modules.utils.translator_utils import get_raw_translation, get_raw_text, format_translations, set_upper_case
from modules.utils.archives import make
from modules.multi_process_block import get_text_items_state

class BatchProcessor:
    def __init__(self):
        self.block_detector_cache = None
        self.inpainter_cache = None
        self.cached_inpainter_key = None
        self.ocr = OCRProcessor()
        
    def skip_save(self, directory: str, timestamp: str, base_name: str, extension: str, archive_bname: str, image: Any) -> None:
        path = os.path.join(directory, f"comic_translate_{timestamp}", "translated_images", archive_bname)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, f"{base_name}_translated{extension}"), image)

    def log_skipped_image(self, directory: str, timestamp: str, image_path: str) -> None:
        with open(os.path.join(directory, f"comic_translate_{timestamp}", "skipped_images.txt"), 'a', encoding='UTF-8') as file:
            file.write(image_path + "\n")

    def get_min_and_max_font_size(self, w):
        max_font_size = int(32 * (w / 960.0) ** 0.5)
        min_font_size = int(16 * (w / 960.0) ** 0.5)

        if w >= 1600:
            if max_font_size > 44:
                max_font_size = 44
            if min_font_size > 20:
                min_font_size = 20
        elif w >= 1280:
            if max_font_size > 40:
                max_font_size = 40
            if min_font_size > 18:
                min_font_size = 18
        elif w >= 720:
            if max_font_size < 32:
                max_font_size = 32
            if min_font_size < 14:
                min_font_size = 14
        elif w >= 600:
            if max_font_size < 24:
                max_font_size = 24
            if min_font_size < 12:
                min_font_size = 12
        else:
            if max_font_size < 18:
                max_font_size = 18
            if min_font_size < 10:
                min_font_size = 10

        return min_font_size, max_font_size

    def get_blk_list(self, settings, image):
        if self.block_detector_cache is None:
            device = 0 if settings.gpu_enabled else 'cpu'
            # self.block_detector_cache = TextBlockDetector(
            #     'models/detection/comic-speech-bubble-detector.pt',
            #     'models/detection/comic-text-segmenter.pt',
            #     'models/detection/manga-text-detector.pt',
            #     device
            # )

            from pathlib import Path
            folder = Path(__file__).parent.parent
            self.block_detector_cache = TextBlockDetector(
                os.path.join(folder, 'models/detection/comic-speech-bubble-detector.pt'),
                os.path.join(folder, 'models/detection/comic-text-segmenter.pt'),
                os.path.join(folder, 'models/detection/manga-text-detector.pt'),
                device
            )

        blk_list = self.block_detector_cache.detect(image)
        return blk_list

    def deal_one_image(self, timestamp, index, total_images, error_msg_arr, image_path, output_path, archive_info,
                       image, blk_list, one_image_state, settings, need_save_path, need_save_image,
                       progress_callback: Callable[[int, int, int, int, bool, str], None] = None,
                       cancel_check: Callable[[], bool] = None, logger=None):
        source_lang = one_image_state['source_lang']
        print('source_lang', source_lang)
        # 用日文代替繁体中文，会有更好的识别效果
        if source_lang == 'Traditional Chinese':
            source_lang = 'Japanese'

        source_lang_en = settings.lang_mapping.get(source_lang, source_lang)
        print('source_lang_en', source_lang_en)
        source_lng_cd = get_language_code(source_lang_en)
        if source_lng_cd is None:
            source_lng_cd = 'en'
        print('source_lng_cd', source_lng_cd)

        target_lang = one_image_state['target_lang']
        print('target_lang', target_lang)
        target_lang_en = settings.lang_mapping.get(target_lang, target_lang)
        print('target_lang_en', target_lang_en)
        trg_lng_cd = get_language_code(target_lang_en)
        if trg_lng_cd is None:
            trg_lng_cd = 'en'
        print('trg_lng_cd', trg_lng_cd)

        base_name = ''
        extension = '.jpg'
        directory = '.'
        if image_path:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            extension = os.path.splitext(image_path)[1]

            if output_path:
                directory = output_path
            else:
                directory = os.path.dirname(image_path)

        archive_bname = ""
        if archive_info:
            for archive in archive_info:
                if image_path in archive['extracted_images']:
                    directory = os.path.dirname(archive['archive_path'])
                    archive_bname = os.path.splitext(os.path.basename(archive['archive_path']))[0]
                    break

        if progress_callback:
            progress_callback(index, total_images, 2, 10, False, "")
        if cancel_check and cancel_check():
            return

        # OCR Processing
        if blk_list:
            cur_t = time.time()
            print(time.time() - cur_t)
            print('------------------ocr.initialize----p--------------')
            print('source_lang', source_lang)
            self.ocr.initialize(settings, source_lang)
            try:
                cur_t = time.time()
                print(time.time() - cur_t)
                print('------------------ocr.process------------------')
                self.ocr.process(image, blk_list)
                source_lang_english = settings.lang_mapping.get(source_lang, source_lang)
                rtl = True if source_lang_english == 'Japanese' else False
                blk_list = sort_blk_list(blk_list, rtl)
                print(time.time() - cur_t)
                cur_t = time.time()
            except Exception as e:
                error_msg = str(e)
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                if progress_callback:
                    progress_callback(index, total_images, 2, 10, False, f"OCR Error: {error_msg}")
                self.log_skipped_image(directory, timestamp, image_path)
                error_msg_arr.append(base_name + f" OCR Error: {error_msg}")
                return
        else:
            print('------------------skip_save------------------')
            self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
            if progress_callback:
                progress_callback(index, total_images, 2, 10, False, "No text blocks detected")
            self.log_skipped_image(directory, timestamp, image_path)
            error_msg_arr.append(base_name + " No text blocks detected")
            return

        if progress_callback:
            progress_callback(index, total_images, 3, 10, False, "")
        if cancel_check and cancel_check():
            return

        # Inpainting
        if self.inpainter_cache is None or self.cached_inpainter_key != settings.inpainter_key:
            device = 'cuda' if settings.gpu_enabled else 'cpu'
            inpainter_key = settings.inpainter_key
            InpainterClass = inpaint_map[inpainter_key]
            self.inpainter_cache = InpainterClass(device)
            self.cached_inpainter_key = inpainter_key

        print('inpainter_cache---%s' % (time.time() - cur_t))
        cur_t = time.time()

        mask = generate_mask(image, blk_list)
        print('generate_mask---%s' % (time.time() - cur_t))
        cur_t = time.time()

        inpaint_input_img = self.inpainter_cache(image, mask, settings.inpaint_config)
        print('inpainter_cache---%s' % (time.time() - cur_t))
        cur_t = time.time()

        inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)
        print('convertScaleAbs---%s' % (time.time() - cur_t))
        cur_t = time.time()

        if progress_callback:
            progress_callback(index, total_images, 4, 10, False, "")
        if cancel_check and cancel_check():
            return

        # Save cleaned image if needed
        if settings.export_inpainted_image:
            path = os.path.join(directory, f"comic_translate_{timestamp}", "cleaned_images", archive_bname)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            cv2.imwrite(os.path.join(path, f" {base_name}_cleaned{extension}"),
                        cv2.cvtColor(inpaint_input_img, cv2.COLOR_BGR2RGB))

        if progress_callback:
            progress_callback(index, total_images, 5, 10, False, "")
        if cancel_check and cancel_check():
            return

        print(time.time() - cur_t)
        cur_t = time.time()
        print('------------------Translation------------------')
        # Translation
        translator = Translator(settings, source_lang, target_lang, logger=logger)
        try:
            translator.translate(blk_list, image, settings.settings_page.llm.extra_context)
        except Exception as e:
            error_msg = str(e)
            self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
            if progress_callback:
                progress_callback(index, total_images, 5, 10, False, f"Translation Error: {error_msg}")
            self.log_skipped_image(directory, timestamp, image_path)
            error_msg_arr.append(base_name + f" Translation Error: {error_msg}")
            return

        print(time.time() - cur_t)
        cur_t = time.time()
        print('-------------Process translation results------------')
        # Process translation results
        entire_raw_text = get_raw_text(blk_list)
        entire_translated_text = get_raw_translation(blk_list)

        try:
            raw_text_obj = json.loads(entire_raw_text)
            translated_text_obj = json.loads(entire_translated_text)

            if (not raw_text_obj) or (not translated_text_obj):
                self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
                if progress_callback:
                    progress_callback(index, total_images, 5, 10, False, "Empty translation result")
                self.log_skipped_image(directory, timestamp, image_path)
                error_msg_arr.append(base_name + " Empty translation result")
                return
        except json.JSONDecodeError as e:
            error_msg = str(e)
            self.skip_save(directory, timestamp, base_name, extension, archive_bname, image)
            if progress_callback:
                progress_callback(index, total_images, 5, 10, False, f"Invalid translation format: {error_msg}")
            self.log_skipped_image(directory, timestamp, image_path)
            error_msg_arr.append(base_name + f" Invalid translation format: {error_msg}")
            return

        # Save text files if needed
        if settings.export_raw_text:
            path = os.path.join(directory, f"comic_translate_{timestamp}", "raw_texts", archive_bname)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"{base_name}_raw.txt"), 'w', encoding='UTF-8') as f:
                f.write(entire_raw_text)

        if settings.export_translated_text:
            path = os.path.join(directory, f"comic_translate_{timestamp}", "translated_texts", archive_bname)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"{base_name}_translated.txt"), 'w', encoding='UTF-8') as f:
                f.write(entire_translated_text)

        if progress_callback:
            progress_callback(index, total_images, 7, 10, False, "")
        if cancel_check and cancel_check():
            return

        print(time.time() - cur_t)
        cur_t = time.time()
        print('-------------Text Rendering------------')
        # Text Rendering
        render_settings = settings.render_settings
        format_translations(blk_list, trg_lng_cd, upper_case=render_settings.upper_case)
        # get_best_render_area(blk_list, image, inpaint_input_img)

        print(time.time() - cur_t)
        cur_t = time.time()
        print('-------------render blk_list------------')

        text_items_state = get_text_items_state(blk_list, render_settings, trg_lng_cd)

        print(time.time() - cur_t)
        cur_t = time.time()
        print('-------------Save rendered image------------')
        # Save rendered image
        sv_pth = None
        if output_path:
            sv_pth = os.path.join(directory, f"{base_name}{extension}")
        else:
            if need_save_path:
                render_save_dir = os.path.join(directory, f"comic_translate_{timestamp}", "translated_images",
                                               archive_bname)
                if not os.path.exists(render_save_dir):
                    os.makedirs(render_save_dir, exist_ok=True)
                sv_pth = os.path.join(render_save_dir, f"{base_name}_translated{extension}")

        im = cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR)
        renderer = ImageSaveRenderer(im)
        viewer_state = one_image_state['viewer_state']
        viewer_state['text_items_state'] = text_items_state
        renderer.add_state_to_image(viewer_state)

        print(time.time() - cur_t)
        cur_t = time.time()
        print('-------------Save rendered image------------')
        output_image = renderer.render_to_image()
        if need_save_image:
            cv2.imwrite(sv_pth, output_image)
            return output_image, sv_pth
        else:
            return output_image, sv_pth


    def process_one_image(self, settings, image, source_lang, target_lang, logger=None):
        h, w, _ = image.shape
        if h * w > 2400 * 3600:
            print('Image too large')
            return False, 'Image too large'
        # 用日文代替繁体中文，会有更好的识别效果
        if source_lang == 'Traditional Chinese':
            source_lang = 'Japanese'

        if h * w > 2400 * 1600 and w > 1600:
            percent = 1600.0 / w
            image = cv2.resize(image, (int(w * percent), int(h * percent)))

        min_font_size, max_font_size = self.get_min_and_max_font_size(w)

        settings.render_settings.max_font_size = max_font_size
        settings.render_settings.min_font_size = min_font_size

        # if w > 1500:
        #     settings.render_settings.max_font_size = int(40 * w / 1500)
        #     settings.render_settings.min_font_size = int(16 * w / 1500)
        # else:
        #     settings.render_settings.max_font_size = 40
        #     settings.render_settings.min_font_size = 16

        blk_list = self.get_blk_list(settings, image)
        if len(blk_list) > 30:
            return False, 'Too much text in image'

        import time
        cur_t = time.time()
        print(time.time() - cur_t)
        print('------------------blk_list------------------')
        print(blk_list)

        one_image_state = {
            'source_lang': source_lang,
            'target_lang': target_lang,
            'viewer_state': {}
        }
        if h > w * 2.1:
            print('合并处理长图')
            result_im, clip_arrs = self.get_combine_clip_arr(image, blk_list)
            tmp_file = 'tmp_%s.jpg'%(time.time())
            cv2.imwrite(tmp_file, result_im)
            combine_image = cv2.imread(tmp_file)
            combine_blk_list = self.get_blk_list(settings, combine_image)
            output_image, save_path = self.deal_one_image(datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p"), 1, 1, [], None, None,
                                               None,
                                               combine_image, combine_blk_list, one_image_state, settings, False, False, None, None, logger)
            for y1, y2, sy1, sy2 in clip_arrs:
                image[sy1+3:sy2-3,:,:] = output_image[y1+3:y2-3,:,:]
            output_image = image
            try:
                os.remove(tmp_file)
            except Exception as ex:
                print(ex)
        else:
            output_image, save_path = self.deal_one_image(datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p"), 1, 1, [], None, None, None,
                                               image, blk_list, one_image_state, settings, False, False, None, None, logger)

        return True, output_image

    def process_images(self,
                       image_files: List[str],
                       image_states: Dict[str, Any],
                       settings,
                       output_path: str = None,
                       archive_info: List[Dict[str, Any]] = None,
                       progress_callback: Callable[[int, int, int, int, bool, str], None] = None,
                       cancel_check: Callable[[], bool] = None, logger=None):
        """
        批量处理图片

        Args:
            image_files: 需要处理的图片文件路径列表
            image_states: 每个图片的状态信息
            settings: 处理设置，包含OCR、翻译、渲染等配置
            output_path: 输出目录（可选）
            archive_info: 压缩包信息（可选）
            progress_callback: 进度回调函数，参数为(当前索引,总数,当前步骤,总步骤数,是否更新名称,错误信息)
            cancel_check: 取消检查函数，返回True表示需要取消处理
        """
        timestamp = datetime.now().strftime("%b-%d-%Y_%I-%M-%S%p")
        total_images = len(image_files)

        import time
        cur_t = time.time()
        print(time.time()-cur_t)

        error_msg_arr = []
        for index, image_path in enumerate(image_files):
            one_image_state = image_states[image_path]
            if progress_callback:
                progress_callback(index, total_images, 0, 10, True, "")
            if cancel_check and cancel_check():
                break

            image = cv2.imread(image_path)
            h, w, _ = image.shape

            if h * w > 3200 * 2400:
                if logger is not None:
                    logger.error(f"翻译超大图片，尺寸: {h}*{w}")

            if h > 10000 or w > 10000:
                print('Image too large')
                error_msg_arr.append(' Image too large')
                continue

            # 大图片将宽度压缩为1600
            if h * w > 2400 * 1600 and w > 1600:
                percent = 1600.0 / w
                w = 1600
                h = int(percent * h)
                image = cv2.resize(image, (w, h))

            min_font_size, max_font_size = self.get_min_and_max_font_size(w)

            settings.render_settings.max_font_size = max_font_size
            settings.render_settings.min_font_size = min_font_size

            # Text Block Detection
            if progress_callback:
                progress_callback(index, total_images, 1, 10, False, "")
            if cancel_check and cancel_check():
                break

            blk_list = self.get_blk_list(settings, image)

            if len(blk_list) > 30:
                if logger is not None:
                    logger.error(f"翻译多文字快图片，文字块数量: {len(blk_list)}")

            if len(blk_list) > 300:
                print('Too much text in one image')
                error_msg_arr.append(' Too much text in one image')
                continue

            print('------------------blk_list------------------')
            print(time.time()-cur_t)
            cur_t = time.time()
            print(blk_list)

            # 判断是否是长图
            if h > w * 3.1:
                print('合并处理长图')
                result_im, clip_arrs = self.get_combine_clip_arr(image, blk_list)
                tmp_file = 'tmp_%s.jpg' % (time.time())
                cv2.imwrite(tmp_file, result_im)
                combine_image = cv2.imread(tmp_file)
                combine_blk_list = self.get_blk_list(settings, combine_image)
                output_image, save_path = self.deal_one_image(timestamp, 1, 1, error_msg_arr, image_path,
                                                  output_path,
                                                  archive_info, combine_image, combine_blk_list, one_image_state, settings, True, False,
                                                  None, cancel_check, logger)
                for y1, y2, sy1, sy2 in clip_arrs:
                    image[sy1 + 3:sy2 - 3, :, :] = output_image[y1 + 3:y2 - 3, :, :]
                cv2.imwrite(save_path, image)
                try:
                    if os.path.isfile(tmp_file):
                        os.remove(tmp_file)
                except Exception as ex:
                    print(ex)
            else:
                self.deal_one_image(timestamp, index, total_images, error_msg_arr, image_path, output_path,
                                                  archive_info, image, blk_list, one_image_state, settings, True, True,
                                                  progress_callback, cancel_check, logger)

        if len(error_msg_arr) > 0:
            return False, '\r\n'.join(error_msg_arr)
        else:
            return True, ''

    def get_y_ranges(self, blk_list, h):
        arr = []
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            y1 -= (y2 - y1) // 2
            if y1 < 0:
                y1 = 0
            y2 += (y2 - y1) // 2
            if y2 > h:
                y2 = h

            cross_arr = [y1, y2]
            left_arr = []
            for a, b in arr[:]:
                if y1<=a<=y2 or y1<=b<=y2 or a<=y1<=b or a<=y2<=b:
                    cross_arr.append(a)
                    cross_arr.append(b)
                else:
                    left_arr.append([a, b])

            left_arr.append([min(cross_arr), max(cross_arr)])
            arr = left_arr

        arr.sort(key=lambda x:x[0])
        print(arr)
        return arr

    def get_combine_clip_arr(self, image, blk_list):
        y_ranges = self.get_y_ranges(blk_list, image.shape[0])

        import numpy as np
        h, w, d = image.shape
        line_width = 6
        zero_arr = np.zeros((line_width, w, d))

        result_im = []
        start_y = 0
        clip_arrs = []
        for y1, y2 in y_ranges:
            result_im.extend(image[y1:y2, :, :])
            clip_arrs.append([start_y, start_y + (y2 - y1), y1, y2])
            result_im.extend(zero_arr)
            start_y += (y2 - y1) + line_width

        return np.array(result_im), clip_arrs

    def run_render_block(self, settings, image, source_lang, target_lang):
        import pickle

        blk_list = self.get_blk_list(settings, image)

        #
        # with open('blk.pkl', 'wb') as fw:
        #     pickle.dump(blk_list, fw)
        #
        mask = generate_mask(image, blk_list)
        InpainterClass = inpaint_map[settings.inpainter_key]
        self.inpainter_cache = InpainterClass('cpu')
        self.cached_inpainter_key = settings.inpainter_key
        inpaint_input_img = self.inpainter_cache(image, mask, settings.inpaint_config)
        inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)
        # with open('inpaint_input_img.pkl', 'wb') as f:
        #     pickle.dump(inpaint_input_img, f)
        #
        im = cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR)
        # with open('im.pkl', 'wb') as f:
        #     pickle.dump(im, f)

        # with open('inpaint_input_img.pkl', 'rb') as fr:
        #     inpaint_input_img = pickle.load(fr)

        # with open('blk.pkl', 'rb') as fr:
        #     blk_list = pickle.load(fr)
        #
        # for blk in blk_list:
        #     print(blk.xyxy)
        #
        # with open('im.pkl', 'rb') as fr:
        #     im = pickle.load(fr)
        #

        # get_best_render_area(blk_list, image, inpaint_input_img)

        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            print(x1, x2, y1, y2)
            xy1 = (x1, y1)
            xy2 = (x2, y2)
            cv2.rectangle(im, xy1, xy2, (0,0,255), thickness=4, lineType=None, shift=None)

            # cv2.putText(im, "test", (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return True, im

    def run_render_combine_clip(self, settings, image, source_lang, target_lang):
        import pickle

        blk_list = self.get_blk_list(settings, image)


        y_ranges = self.get_y_ranges(blk_list, image.shape[0])

        im = image

        #
        # with open('blk.pkl', 'wb') as fw:
        #     pickle.dump(blk_list, fw)
        #
        # mask = generate_mask(image, blk_list)
        # InpainterClass = inpaint_map[settings.inpainter_key]
        # self.inpainter_cache = InpainterClass('cpu')
        # self.cached_inpainter_key = settings.inpainter_key
        # inpaint_input_img = self.inpainter_cache(image, mask, settings.inpaint_config)
        # inpaint_input_img = cv2.convertScaleAbs(inpaint_input_img)
        # with open('inpaint_input_img.pkl', 'wb') as f:
        #     pickle.dump(inpaint_input_img, f)
        #
        # im = cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR)
        # with open('im.pkl', 'wb') as f:
        #     pickle.dump(im, f)


        # with open('inpaint_input_img.pkl', 'rb') as fr:
        #     inpaint_input_img = pickle.load(fr)

        # with open('blk.pkl', 'rb') as fr:
        #     blk_list = pickle.load(fr)
        #
        # for blk in blk_list:
        #     print(blk.xyxy)
        #
        # with open('im.pkl', 'rb') as fr:
        #     im = pickle.load(fr)
        #
        # h, w, _ = im.shape
        # print(h)
        #
        # y_ranges = self.get_y_ranges(blk_list, h)

        # get_best_render_area(blk_list, image, inpaint_input_img)

        # for blk in blk_list:
        #     x1, y1, x2, y2 = blk.xyxy
        #     xy1 = (x1, y1)
        #     xy2 = (x2, y2)
        #     cv2.rectangle(im, xy1, xy2, (0,0,255), thickness=4, lineType=None, shift=None)
        #
        #     # cv2.putText(im, "test", (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        import numpy as np
        h, w, d = im.shape
        line_width = 6
        zero_arr = np.zeros((line_width, w, d))

        result_im = []
        start_y = 0
        clip_arrs = []
        for y1, y2 in y_ranges:
            result_im.extend(im[y1:y2, :, :])
            clip_arrs.append([start_y, start_y+(y2-y1), y1, y2])
            result_im.extend(zero_arr)
            start_y += (y2-y1) + line_width

        print(clip_arrs)

        return True, np.array(result_im), clip_arrs


if __name__ == '__main__':
    BatchProcessor().process_images()