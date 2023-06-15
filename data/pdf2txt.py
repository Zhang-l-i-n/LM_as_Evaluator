# from pdfminer.high_level import extract_text
#
# def extract_text_from_pdf(file_path):
#     return extract_text(file_path)
#
# text = extract_text_from_pdf('./课本/生物/生物技术与工程.pdf')
#
#
# # 保存为纯文本文件
# with open('biography5.txt', 'w', encoding='utf-8') as f:
#     f.write(text)
import fitz # PyMuPDF

doc = fitz.open('./课本/生物/生物技术与工程.pdf')

for page in doc:
    # 获取页面的宽度和高度
    width, height = page.rect.width, page.rect.height

    # 定义要保留的页面部分，例如保留中间的70%，则上下各忽略15%
    margin_bottom = height * 0.90

    # 创建一个新的矩形，忽略顶部和底部
    rect = fitz.Rect(0, 0, width, margin_bottom)

    # 获取该区域的文本
    text = page.get_textbox(rect)
    print(text)

    with open('biography5.txt', 'a', encoding='utf-8') as f:
        f.write(text)
