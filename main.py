import layoutparser as lp

pdf_file = "A_decoder-only_foundation_model_for_time-series.pdf"
pages = lp.load_pdf(pdf_file, load_words=True, extra_attrs=["fontname", "size"])
model = lp.models.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')

for page in pages:
    layout = model.detect(page)
    for block in layout:
        print(block.type, block.coordinates)
