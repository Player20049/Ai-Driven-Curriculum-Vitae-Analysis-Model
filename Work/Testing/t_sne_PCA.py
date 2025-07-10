'''
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# -------- 1. Load PDF and extract text --------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

cv_path = "C:/Users/zeiad/Python/vs_code/LSBU/Beng Project/uploaded_files/1-1Software Engineer CV.pdf"  # ← your CV file here
jd_path = "C:/Users/zeiad/Python/vs_code/LSBU/Beng Project/uploaded_jds/1-Software Engineer JD.pdf"  # ← your JD file here

cv_text = extract_text_from_pdf(cv_path)
jd_text = extract_text_from_pdf(jd_path)

# -------- 2. Load model and encode embeddings --------
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

texts = [jd_text, cv_text]
embeddings = model.encode(texts, normalize_embeddings=True)

# -------- 3. Apply PCA for 2D visualization --------
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# -------- 4. Plot --------
plt.figure(figsize=(8, 6))
plt.scatter(reduced[0, 0], reduced[0, 1], c='blue', marker='^', label='Job Description', s=200)
plt.scatter(reduced[1, 0], reduced[1, 1], c='red', marker='o', label='CV', s=200)

# Add line showing semantic distance
plt.plot([reduced[0, 0], reduced[1, 0]], [reduced[0, 1], reduced[1, 1]], linestyle='--', color='gray')

plt.title("PCA Visualization of CV and JD Embeddings (all-mpnet-base-v2)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#---------------------------------------

import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# --------- CONFIG ---------
cv_dir = "CVS"
jd_dir = "JDS"
model_name = 'sentence-transformers/all-mpnet-base-v2'

# --------- FUNCTIONS ---------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return " ".join([page.get_text() for page in doc]).strip()

# --------- LOAD TEXTS ---------
cv_texts = []
jd_texts = []
cv_labels = []
jd_labels = []

for filename in os.listdir(cv_dir):
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(os.path.join(cv_dir, filename))
        cv_texts.append(text)
        cv_labels.append(f"CV: {filename}")

for filename in os.listdir(jd_dir):
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(os.path.join(jd_dir, filename))
        jd_texts.append(text)
        jd_labels.append(f"JD: {filename}")

all_texts = jd_texts + cv_texts
all_labels = jd_labels + cv_labels

# --------- EMBEDDING ---------
model = SentenceTransformer(model_name)
embeddings = model.encode(all_texts, normalize_embeddings=True)

# --------- PCA ---------
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# --------- PLOT ---------
plt.figure(figsize=(10, 7))
# Plot JDs
plt.scatter(reduced[:len(jd_texts), 0], reduced[:len(jd_texts), 1], c='blue', marker='^', label='Job Descriptions (JDs)', s=100)
# Plot CVs
plt.scatter(reduced[len(jd_texts):, 0], reduced[len(jd_texts):, 1], c='red', marker='o', label='CVs', s=60)

# Optional: annotate a few points
for i, label in enumerate(all_labels):
    if i < 3 or i > len(jd_texts):  # Annotate first few JDs and all CVs
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8)

plt.title("PCA Visualization of CV and JD Embeddings")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

'''
#---------------------------------------------------------------

import os # For file path operations and directory listing
import fitz  # PyMuPDF for reading text from PDF files
from sentence_transformers import SentenceTransformer # For generating semantic embeddings
from sklearn.decomposition import PCA # For dimensionality reduction
import matplotlib.pyplot as plt # For plotting
import numpy as np # For numerical operations
from matplotlib.lines import Line2D # For custom legend elements
from adjustText import adjust_text # For automatically adjusting text labels in plots

# --------- CONFIG ---------
cv_dir = "CVS" # Directory containing CV PDF files
jd_dir = "JDS" # Directory containing JD PDF files
model_name = 'sentence-transformers/all-mpnet-base-v2' # Pretrained sentence transformer model

# --------- FUNCTIONS ---------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path) # Open the PDF file
    return " ".join([page.get_text() for page in doc]).strip()  # Join text from all pages

# --------- LOAD TEXTS ---------
cv_texts = [] # List to store extracted CV texts
jd_texts = [] # List to store extracted JD texts
cv_labels = [] # List to store CV filenames
jd_labels = [] # List to store JD filenames

# Loop through all files in the JD directory
for filename in os.listdir(cv_dir):
    if filename.endswith(".pdf"): # Only process PDF files
        text = extract_text_from_pdf(os.path.join(cv_dir, filename)) # Extract text
        cv_texts.append(text) # Append text to CV list
        cv_labels.append(filename) # Append filename to label list

for filename in os.listdir(jd_dir):
    if filename.endswith(".pdf"): # Only process PDF files
        text = extract_text_from_pdf(os.path.join(jd_dir, filename)) # Extract text
        jd_texts.append(text) # Append text to CV list
        jd_labels.append(filename) # Append filename to label list

all_texts = jd_texts + cv_texts # Combine JD and CV texts for embedding

# --------- EMBEDDING ---------
model = SentenceTransformer(model_name) # Load sentence transformer model
embeddings = model.encode(all_texts, normalize_embeddings=True) # Generate and normalize embeddings

# --------- PCA ---------
pca = PCA(n_components=2) # Initialize PCA with 2 components
reduced = pca.fit_transform(embeddings) # Reduce embedding dimensions to 2D

# --------- PLOT ---------
plt.figure(figsize=(12, 8)) # Create a new plot with specified size

texts = [] # List to hold text labels for adjustment

# Plot and label JDs
for i in range(len(jd_texts)):
    x, y = reduced[i] # Get PCA coordinates
    plt.scatter(x, y, c='blue', marker='^', s=100) # Plot JD point as blue triangle
    texts.append(plt.text(x, y, f"JD{i+1}", fontsize=9)) # Add label

# Plot and label CVs
for i in range(len(cv_texts)): 
    x, y = reduced[len(jd_texts) + i] # Offset index to get CV points
    plt.scatter(x, y, c='red', marker='o', s=60) # Plot CV point as red circle
    texts.append(plt.text(x, y, f"CV{i+1}", fontsize=9)) # Add label

# Auto-adjust to avoid label overlaps
adjust_text(
    texts, # List of text objects
    only_move={'points': 'y', 'texts': 'xy'}, # Restrict movement direction
    arrowprops=dict(arrowstyle="-", color='gray', lw=0.5) # Arrow style for displaced labels
)

# Legend for shapes/colors only
legend_elements = [
    Line2D([0], [0], marker='^', color='w', label='Job Descriptions (JDs)', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='CVs', markerfacecolor='red', markersize=8)
]
plt.legend(handles=legend_elements, loc='upper right') # Display legend in top-right

plt.title("PCA Visualization of CV and JD Embeddings") # Display legend in top-right
plt.xlabel("PCA Dimension 1") # Label X-axis
plt.ylabel("PCA Dimension 2") # Label Y-axis
plt.grid(True) # Enable grid lines
plt.tight_layout() # Adjust layout to fit everything nicely
plt.savefig("semantic_plot.png", dpi=300) # Save plot as high-res PNG
plt.show() # Display the plot

# --------- PRINT LEGEND ---------
print("\n JD Legend:")
for i, filename in enumerate(jd_labels):
    print(f"JD{i+1}: {filename}")

print("\n CV Legend:") # Print filename legend for CVs
for i, filename in enumerate(cv_labels):
    print(f"CV{i+1}: {filename}")
