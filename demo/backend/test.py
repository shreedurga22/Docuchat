import fitz
print(fitz.__version__)
doc = fitz.open("uploads/test.pdf")  # replace with a real PDF path you have
print(len(doc))  # prints number of pages
doc.close()
