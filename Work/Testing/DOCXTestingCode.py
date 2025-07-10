import docx  # To read .docx documents
import pytesseract  # OCR for reading text from images
from PIL import Image  # To process images from DOCX files
import io  # To handle image data in memory
from tkinter import Tk, filedialog  # To allow us to select the file on our local machine

def upload_and_extract_docx():
    # Open a file dialog to select the DOCX
    Tk().withdraw()  # Hide the root Tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("Word Documents", "*.docx")])

    # Debugging prints text if no file is uploaded
    if not file_path:
        print("No file selected.")
        return

    # Open and read the DOCX file
    doc = docx.Document(file_path)  # Calls the file path and opens it up
    text = ""

    # Extract text from paragraphs
    for para in doc.paragraphs:
        text += para.text + " "

    # Extract text from images (OCR)
    for rel in doc.part.rels:
        if "image" in doc.part.rels[rel].target_ref:
            image_data = doc.part.rels[rel].target_part.blob
            img = Image.open(io.BytesIO(image_data))

            # Perform OCR on the image
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + " "  # Append OCR text to main text content

    # Print the extracted content
    print("\nExtracted Text from DOCX:\n")
    print(text)

# Run the function
upload_and_extract_docx()



# PyMuPDF version: 1.24.10  # Used for reading and extracting text from PDF documents
# pytesseract version: 5.4.0.20240606  # OCR engine used to extract text from images inside PDFs and DOCX files
# Pillow version: 10.4.0  # Used to process images extracted from PDFs and DOCX for OCR
# io module: Built-in, no version is required here  # Used for handling image data in memory when extracting images from documents
# tkinter module: Built-in, no version is required here  # Used for creating a file dialog to allow users to select PDF/DOCX files
# python-docx version: 1.1.2  # Used for reading and extracting text from DOCX (Microsoft Word) documents
