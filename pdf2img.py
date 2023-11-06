import office
import os

if not os.path.exists("./test_img"):
    os.makedirs("./test_img")
office.pdf.pdf2imgs("./test_pdf", "./test_img")
