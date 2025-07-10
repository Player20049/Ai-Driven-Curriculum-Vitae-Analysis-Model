'''
import fitz  # PyMuPDF To read the pdf document and all of its content 
from tkinter import Tk, filedialog # To allow us to select the file on our local machine 

def upload_and_extract_pdf():
    # Open a file dialog to select the PDF
    Tk().withdraw()  # Hide the root Tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])

    #debugging prints text if no file is uploaded 
    if not file_path:
        print("No file selected.")
        return

    # Open and read the PDF file
    doc = fitz.open(file_path) #calls the file path and opens it up
    text = "" 

    # Extract text from each page
    for page in doc:
        text += page.get_text("text") 

    # Print the extracted content
    print("Extracted Text from PDF:")
    print(text)

# Run the function
upload_and_extract_pdf()


#Code used for pdf Text extraction and OCR check for scanned documents 
import fitz  # PyMuPDF To read the pdf document and all of its content
import pytesseract  # OCR for reading text from images
from PIL import Image  # To process images from PDFs
import io  # To handle image data in memory
from tkinter import Tk, filedialog  # To allow us to select the file on our local machine

def upload_and_extract_pdf():
    # Open a file dialog to select the PDF
    Tk().withdraw()  # Hide the root Tkinter window
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])

    # Debugging prints text if no file is uploaded
    if not file_path:
        print("No file selected.")
        return

    # Open and read the PDF file
    doc = fitz.open(file_path)  # Calls the file path and opens it up
    text = ""  

    # Extract text from each page
    for page in doc:
        # Extract text from normal PDF content
        text += page.get_text("text") + " "

        # Extract text from images (OCR)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            img = Image.open(io.BytesIO(image_data))

            # Perform OCR on the image
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + " "  # Append OCR text to main text content

    # Print the extracted content
    print("\nExtracted Text from PDF:\n")
    print(text)

# Run the function
upload_and_extract_pdf()

# PyMuPDF version: 1.24.10  # Used for reading and extracting text from PDF documents
# pytesseract version: 5.4.0.20240606  # OCR engine used to extract text from images inside PDFs and DOCX files
# Pillow version: 10.4.0  # Used to process images extracted from PDFs and DOCX for OCR
# io module: Built-in, no version is required here  # Used for handling image data in memory when extracting images from documents
# tkinter module: Built-in, no version is required here  # Used for creating a file dialog to allow users to select PDF/DOCX files
# python-docx version: 1.1.2  # Used for reading and extracting text from DOCX (Microsoft Word) documents
'''
import fitz  # PyMuPDF
import pytesseract
import PIL  # Pillow
import io
import tkinter
import docx

# Print the versions
print("PyMuPDF version:", fitz.__version__)
print("pytesseract version:", pytesseract.get_tesseract_version())  # Requires Tesseract installed
print("Pillow version:", PIL.__version__)
print("io module: Built-in, no version required")
print("tkinter module: Built-in, no version required")
print("python-docx version:", docx.__version__)


