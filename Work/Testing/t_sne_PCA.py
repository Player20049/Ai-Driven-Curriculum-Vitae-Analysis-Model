
#This Figure can be found in "Final Report" it is Figure 13 and is under this directory exactly Ai-Driven-Curriculum-Vitae-Analysis-Model/Work

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
