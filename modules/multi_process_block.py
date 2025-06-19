from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from modules.rendering.maga_render import pyside_word_wrap

from enum import Enum

class PColor:
    def __init__(self, r=0, g=0, b=0, a=255):
        self._r = min(max(r, 0), 255)
        self._g = min(max(g, 0), 255)
        self._b = min(max(b, 0), 255)
        self._a = min(max(a, 0), 255)

    @classmethod
    def from_hex(cls, hex_str):
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            return cls(r, g, b)
        elif len(hex_str) == 8:
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)
            a = int(hex_str[6:8], 16)
            return cls(r, g, b, a)
        raise ValueError("无效的十六进制颜色值")

    def to_hex(self):
        return f"#{self._r:02x}{self._g:02x}{self._b:02x}"

    def to_rgba(self):
        return (self._r, self._g, self._b, self._a)

    def to_rgb(self):
        return (self._r, self._g, self._b)

    def to_normalized_rgba(self):
        return (self._r / 255.0, self._g / 255.0, self._b / 255.0, self._a / 255.0)

    def to_normalized_rgb(self):
        return (self._r / 255.0, self._g / 255.0, self._b / 255.0)

    @property
    def red(self):
        return self._r

    @property
    def green(self):
        return self._g

    @property
    def blue(self):
        return self._b

    @property
    def alpha(self):
        return self._a

class OutlineType(Enum):
    Full_Document = 'full_document'
    Selection = 'selection'

class OutlineInfo:
    start: int
    end: int
    color: PColor
    width: float
    type: OutlineType

    def __init__(self, start, end, color, width, type):
        self.start = start
        self.end = end
        self.color = color
        self.width = width
        self.type = type

def process_block(blk, render_settings, trg_lng_cd):
    """处理单个文本块的线程函数"""
    x1, y1, width, height = blk.xywh
    # print('process_block-xywh-%s,%s,%s,%s' % (x1, y1, width, height))

    translation = blk.translation
    if not translation or len(translation) == 1:
        return None

    if trg_lng_cd in {'zh-CN', 'zh-TW', 'zh', 'ja'}:
        width *= 1.2

    translation, font_size, msg_width, msg_height = pyside_word_wrap(
        translation,
        render_settings.font_family,
        width, height,
        float(render_settings.line_spacing),
        float(render_settings.outline_width),
        render_settings.bold,
        render_settings.italic,
        render_settings.underline,
        render_settings.alignment,
        render_settings.direction,
        render_settings.max_font_size,
        render_settings.min_font_size,
        break_long_words=trg_lng_cd in {'zh-CN', 'zh-TW', 'zh', 'ja'}
    )

    # print('alignment-%s' % render_settings.alignment)
    # print('alignment-w-h-%s-%s' % (msg_width, msg_height))

    # print('---pyside_word_wrap---%s---%s-----%s'%(font_size, msg_width, msg_height))

    if any(lang in trg_lng_cd.lower() for lang in ['zh', 'ja', 'th']):
        translation = translation.replace(' ', '')

    y1 += (height - msg_height) * 0.3
    x1 += (width - msg_width) * 0.5

    return {
        'text': translation,
        'font_family': render_settings.font_family,
        'font_size': font_size,
        'text_color': render_settings.color,
        'alignment': render_settings.alignment,
        'line_spacing': float(render_settings.line_spacing),
        'outline_color': render_settings.outline_color,
        'outline_width': float(render_settings.outline_width),
        'bold': render_settings.bold,
        'italic': render_settings.italic,
        'underline': render_settings.underline,
        'position': (x1, y1),
        'rotation': blk.angle,
        'scale': 1.0,
        'transform_origin': blk.tr_origin_point,
        'width': width,
        'direction': render_settings.direction,
        'selection_outlines': [OutlineInfo(0, len(translation),
                                           render_settings.outline_color,
                                           float(render_settings.outline_width),
                                           OutlineType.Full_Document)] if render_settings.outline else []
    }

def get_text_items_state(blk_list, render_settings, trg_lng_cd):
    # 使用进程池并行处理文本块
    max_workers = max(1, cpu_count() - 1)  # 保留一个核心给系统
    text_items_state = []
    print('---------------max_workers-------------', max_workers)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到进程池
        future_to_block = {executor.submit(process_block, blk, render_settings, trg_lng_cd): blk 
                          for blk in blk_list}
        
        # 收集处理结果
        for future in future_to_block:
            try:
                result = future.result()
                if result:
                    text_items_state.append(result)
            except Exception as e:
                print(f"处理文本块时发生错误: {str(e)}")
                continue
    
    return text_items_state