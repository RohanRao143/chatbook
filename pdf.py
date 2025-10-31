from PyPDF2 import PdfReader

import pymupdf
import re

doc = pymupdf.open("MRohanRaoResumeProfile2025-1.pdf.pdf") # open a document
# out = open("output.txt", "wb") # create a text output
# for page in doc: # iterate the document pages
#     text = page.get_text().encode("utf8") # get plain text (is in UTF-8)

#     text = str(text).split("\\n")
#     # re.sub(r'[^a-zA-Z0-9]', ' ', _.strip()) 
#     data = [_.strip() for _ in text if len(_) > 4]
#     print("\n +++++++++++++++++++", data)











sentences = []

for page in doc: # iterate the document pages
    text = page.get_text().encode("utf8") # get plain text (is in UTF-8)

    text = str(text).split("\\n")
    # re.sub(r'[^a-zA-Z0-9]', ' ', _.strip()) 
    data = [_.strip() for _ in text if len(_) > 4]

    if len(data) > 0:
        sentences += data

print(sentences)
    # out.write(text) # write text of page
    # out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
# out.close()


# reader = PdfReader("MRohanRaoResumeProfile2025-1.pdf.pdf")
# number_of_pages = len(reader.pages)

# for index in range(number_of_pages):
#     page = reader.pages[index]
#     text = page.extract_text()

#     print("\\n\n", len(text), "===", "------", text.split("\n"))

