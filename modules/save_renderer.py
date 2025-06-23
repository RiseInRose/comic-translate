import cv2, os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any

class ImageSaveRenderer:
    def __init__(self, cv2_image):
        self.cv2_image = cv2_image
        self.image = self.cv2_to_pil(cv2_image)
        self.draw = None
        self.text_items = []

    def cv2_to_pil(self, cv2_img):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)

    def pil_to_cv2(self, pil_image):
        # Convert PIL image to cv2 format
        numpy_image = np.array(pil_image)
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    def add_state_to_image(self, state):
        self.text_items = []
        for text_block in state.get('text_items_state', []):
            self.text_items.append(text_block)

    def render_text(self, text_block: Dict[str, Any], scale_factor: float = 2.0):
        # Create a font object
        font = ImageFont.truetype(text_block['font_family'], int(text_block['font_size'] * scale_factor))

        # Calculate text position
        pos_x, pos_y = text_block['position']
        
        # Apply transformations
        if text_block.get('transform_origin'):
            origin_x, origin_y = text_block['transform_origin']
            # In PIL, we handle rotation around a point by adjusting the text position
            if text_block.get('rotation', 0) != 0:
                angle = text_block['rotation']
                # TODO: Implement complex rotation with origin point if needed

        # Draw text outline if specified
        if text_block.get('outline_width', 0) > 0:
            outline_width = int(text_block['outline_width'] * scale_factor)
            outline_color = text_block['outline_color']
            # Draw outline by offsetting text in all directions
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    self.draw.text(
                        (pos_x * scale_factor + dx, pos_y * scale_factor + dy),
                        text_block['text'],
                        font=font,
                        fill=outline_color,
                        align=text_block['alignment']
                    )

        # Draw main text
        self.draw.text(
            (pos_x * scale_factor, pos_y * scale_factor),
            text_block['text'],
            font=font,
            fill=text_block['text_color'],
            align=text_block['alignment']
        )

    def render_to_image(self):
        # Create a high-resolution image
        scale_factor = 2
        width, height = self.image.size
        scaled_image = Image.new('RGB', (width * scale_factor, height * scale_factor))
        scaled_image.paste(self.image.resize((width * scale_factor, height * scale_factor), Image.Resampling.LANCZOS))
        
        # Create a drawing context for the scaled image
        self.draw = ImageDraw.Draw(scaled_image)
        
        # Render all text items
        for text_block in self.text_items:
            self.render_text(text_block, scale_factor)
        
        # Scale back down to original size with antialiasing
        final_image = scaled_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert back to cv2 format
        return self.pil_to_cv2(final_image)

    def save_image(self, output_path):
        final_image = self.render_to_image()
        cv2.imwrite(output_path, final_image)

    @staticmethod
    def add_watermark(image, text='mangatranslate.com'):
        h, w, _ = image.shape

        # 将OpenCV图像转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 创建一个透明图层用于绘制水印
        watermark = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)

        src_folder = Path(__file__).parent.parent
        font_path = os.path.join(src_folder, 'fonts/Arial-Unicode-Regular.ttf')

        # 使用指定的字体文件
        font_size = 24 if w < 1000 else w // 40  # 设置较大的字号
        try:
            # 使用项目中的anime_ace_3.ttf字体
            font = ImageFont.truetype(font_path, font_size)
        except:
            # 如果找不到字体文件，使用默认字体
            font = ImageFont.load_default()

        # 获取文字尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 创建一个临时图像来绘制文字
        temp_img = Image.new('RGBA', (text_width + 50, text_height + 50), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_img)

        # 绘制白色阴影（偏移几个像素）
        shadow_offset = 2
        shadow_alpha = int(255 * 0.95)  # 阴影
        temp_draw.text((25 + shadow_offset, 25 + shadow_offset), text, font=font,
                       fill=(255, 255, 255, shadow_alpha))  # 白色阴影
        temp_draw.text((25 - shadow_offset, 25 + shadow_offset), text, font=font, fill=(255, 255, 255, shadow_alpha))
        temp_draw.text((25 + shadow_offset, 25 - shadow_offset), text, font=font, fill=(255, 255, 255, shadow_alpha))
        temp_draw.text((25 - shadow_offset, 25 - shadow_offset), text, font=font, fill=(255, 255, 255, shadow_alpha))

        # 在临时图像上绘制黑色文字（主文字），多次绘制实现加粗效果，设置透明度
        text_alpha = int(255 * 0.7)  # 设置0.3的透明度
        temp_draw.text((25, 25), text, font=font, fill=(0, 0, 0, text_alpha))
        # temp_draw.text((25 + 1, 25), text, font=font, fill=(0, 0, 0, text_alpha))  # 右偏移1像素
        # temp_draw.text((25, 25 + 1), text, font=font, fill=(0, 0, 0, text_alpha))  # 下偏移1像素
        # temp_draw.text((25 + 1, 25 + 1), text, font=font, fill=(0, 0, 0, text_alpha))  # 右下偏移1像素

        # 不再旋转文字，直接使用原图像
        rotated_text = temp_img

        # 计算粘贴位置（右下角）
        paste_x = w - rotated_text.width - 5  # 减少右边距，更靠近右边
        paste_y = h - rotated_text.height - 5  # 减少下边距，更靠近底部

        # 确保不超出边界
        paste_x = max(0, paste_x)
        paste_y = max(0, paste_y)

        # 将旋转后的文字粘贴到水印图层
        watermark.paste(rotated_text, (paste_x, paste_y), rotated_text)

        # 将水印合成到原图像
        pil_image = pil_image.convert('RGBA')
        combined = Image.alpha_composite(pil_image, watermark)

        # 转换回OpenCV格式
        result_image = cv2.cvtColor(np.array(combined.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        return result_image