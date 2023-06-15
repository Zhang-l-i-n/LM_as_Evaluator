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

# docs = ['./课本/历史/中外历史纲要（上）.pdf', './课本/历史/中外历史纲要（下）.pdf', './课本/历史/国家制度与社会治理.pdf', './课本/历史/经济与社会生活.pdf', './课本/历史/文化交流与传播.pdf']
docs = ['./课本/生物/分子与细胞.pdf', './课本/生物/遗传与进化.pdf', './课本/生物/稳态与调节.pdf', './课本/生物/生物与环境.pdf', './课本/生物/生物技术与工程.pdf']

for i in range(len(docs)):
    doc = fitz.open(docs[i])

    for page in doc:
        width, height = page.rect.width, page.rect.height
        margin_bottom = height * 0.90
        rect = fitz.Rect(0, 0, width, margin_bottom)

        # 获取该区域的文本
        text = page.get_textbox(rect)
        print(text)

        with open('text/biography'+str(i+1)+'.txt', 'a', encoding='utf-8') as f:
            f.write(text + '\n')
