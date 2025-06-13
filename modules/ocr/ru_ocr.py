from paddleocr import PaddleOCR

# 初始化OCR模型（指定俄语语言包）
ocr = PaddleOCR(
    lang='ru',  # 俄语语言包
    use_gpu=True,  # 启用GPU加速
    use_angle_cls=True,  # 启用方向分类（应对倾斜文本）
    rec_model_dir='./models/ru_rec',  # 自定义俄语识别模型（可选）
    det_model_dir='./models/ru_det'   # 自定义俄语检测模型（可选）
)

# 读取图片并识别
image_path = '/Users/mac/Documents/manga/WX20250612-202227@2x.png'
result = ocr.ocr(image_path, cls=True)

# 输出识别结果
for line in result:
    text = line[1][0]  # 提取文本
    confidence = line[1][1]  # 置信度
    print(f"文本: {text}, 置信度: {confidence:.2f}")