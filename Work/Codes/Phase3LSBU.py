#This code Should work with the "LSBUPhase3.html template they everything worked last check was on 27/01/2025"
from flask import Flask, render_template, request, jsonify # Import necessary Flask modules for building the web application
from sentence_transformers import SentenceTransformer, util 
import fitz  # PyMuPDF for PDF handling and text extraction from PDF
import os # Import os for setting environment variables and handling file system operations
from collections import Counter  #Import Counter from collections to count keyword occurrences
import docx

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress TensorFlow warnings
app = Flask(__name__, template_folder='templates') # Initialize Flask app and specify the folder for HTML templates
model = SentenceTransformer('all-mpnet-base-v2') # Load Sentence Transformer model

def extract_text_from_file(file):
    # Define a function that extracts text from an uploaded file
    try:
        # Get the filename and convert it to lowercase for consistent format checking
        filename = file.filename.lower()
        
        if filename.endswith('.pdf'):
            # If the file is a PDF, extract the text using PyMuPDF (fitz)
            doc = fitz.open(stream=file.read(), filetype="pdf") # Open the PDF file from memory stream
            return " ".join([page.get_text() for page in doc]) # Extract text from each page and join them into a single string
        
        elif filename.endswith('.docx'):
            # If the file is a DOCX, extract the text using python-docx
            doc = docx.Document(file) # Open the DOCX file using the python-docx library
            # Extract text from each paragraph and join them into a single string separated by newlines
            return "\n".join([para.text for para in doc.paragraphs])
        
        else:
            # If the file type is not supported, raise a ValueError
            raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")
    
    except Exception as e:
        # Catch any exceptions that occur during file reading or text extraction
        print(f"Error extracting text: {str(e)}") # Print the error message to the console
        raise # Re-raise the exception so it can be handled by the calling function

# Helper function to extract keywords
def extract_keywords(text):
    """Extracts keywords from text using simple tokenization."""
    words = text.split() # Split text into individual words using spaces
    common_words = set(["and", "or", "the", "a", "in", "to", "of", "for", "with", "on", "by", "as", "at", "is"])  # Stop words
    keywords = [word.strip().lower() for word in words if word.isalpha() and word.lower() not in common_words] # Use Counter to count the frequency of each keyword
    return Counter(keywords) 

# Home Route
@app.route('/')
def home():
    return render_template('LSBUPhase3.html') # Render the main HTML template LSBUPhase3.html

# Route for JD and CV analysis
@app.route('/option3', methods=['GET', 'POST'])
def option3():
    if request.method == 'POST': # If the request method is POST
        jd_files = request.files.getlist('jd_files') # Get list of uploaded JD files
        cv_files = request.files.getlist('cv_files') # Get list of uploaded CV files

        # Error handling if no JD files are uploaded
        if not jd_files:
            return jsonify({'error': 'Please upload at least one JD file'}), 400
        # Error handling if no CV files are uploaded
        if not cv_files:
            return jsonify({'error': 'Please upload at least one CV file'}), 400

        # Extract text from JDs and CVs using the combined function
        jd_texts = [extract_text_from_file(jd) for jd in jd_files]
        cv_texts = [extract_text_from_file(cv) for cv in cv_files]

        # Encode JDs and CVs using the SentenceTransformer model
        # Encode JDs and CVs using the SentenceTransformer model
        jd_embeddings = [model.encode(jd_text, convert_to_tensor=True) for jd_text in jd_texts] # Convert JD text into embeddings using the Sentence Transformer model
        cv_embeddings = [model.encode(cv_text, convert_to_tensor=True) for cv_text in cv_texts] # Convert CV text into embeddings using the Sentence Transformer model

        # Calculate similarity scores for each JD-CV pair
        results = [] # Create an empty list to store results
        for jd_idx, jd_embedding in enumerate(jd_embeddings): # Loop through each JD embedding
            jd_keywords = extract_keywords(jd_texts[jd_idx]) # Extract keywords from the JD text
            for cv_idx, cv_embedding in enumerate(cv_embeddings): # Loop through each CV embedding
                score = util.pytorch_cos_sim(jd_embedding, cv_embedding).item() # Compute cosine similarity between JD and CV embeddings
                cv_keywords = extract_keywords(cv_texts[cv_idx]) # Extract keywords from the CV text
                matched_keywords = set(jd_keywords.keys()) & set(cv_keywords.keys()) # Find matched keywords between JD and CV
                results.append({
                    "jd_name": jd_files[jd_idx].filename, # JD file name
                    "cv_name": cv_files[cv_idx].filename, # CV file name
                    "similarity_score": round(score, 4), # Round similarity score to 4 decimal places
                    "matched_keywords": list(matched_keywords), # List of matched keywords
                    "total_matches": len(matched_keywords) # Number of matched keywords
                })
        return jsonify({'results': results}) #Return Results as JSON to Frontend
    return render_template('phase3HTML.html') # If the request method is GET, render the HTML template
# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True, port=5001) # Start the Flask app in debug mode on port 5001



