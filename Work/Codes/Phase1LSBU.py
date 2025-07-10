#This code Should work with the "LSBUPhase1.html template they everything worked last check was on 27/01/2025"
import re # For regular expressions used in skill extraction and keyword pattern matching
import fitz  # PyMuPDF for PDF handling
import docx  # for handling DOCX files
import os # For file path handling, enviroment variables and working directory checks
from PIL import Image # To handle image formats if OCR is needed 
#from PyQt5.QtWidgets  # PyQt5 import 
from fuzzywuzzy import fuzz, process # Import fuzzy matching functions from the fuzzywuzzy library
from pymongo import MongoClient # To connect to MongoDB for storing CVs 
from flask import Flask, render_template, request, jsonify, send_file, abort, url_for, redirect 
# Flask framework used fro setting up backend API
import hashlib # For hasjhing CV file content 
import smtplib #  For sending emails
from email.mime.multipart import MIMEMultipart # For building multipart email 
from email.mime.text import MIMEText # to define the body of the email 
import schedule # For setting scheduled tasks
import time # used with scheduling and timing events
import requests  # Added for ESCO API integration
print("Current Working Directory:", os.getcwd())


# Global variable to store dynamically generated skills from ESCO API
extracted_skills_ESCO = []

app = Flask(__name__)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['cv_analysis2_db']  # Database name
cv_collection = db['cvs2']  # Collection name

# Directory to save uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploaded_files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists

# Preprocessing: clean the text (lowercase, remove special characters)
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

# Function to save CV data into MongoDB in an organized way
def save_cv_to_mongodb(filename, structured_data):
    print("Saving CV to MongoDB:", structured_data)  # Print the structured data for debugging purposes
    cv_hash = hashlib.sha256(structured_data['cv_text'].encode()).hexdigest() # Generate a unique hash of the CV text using SHA-256 to detect duplicates
    existing_cv = cv_collection.find_one({"cv_hash": cv_hash}) # Check if a CV with the same hash already exists in the database   
    # If a CV with the same hash exists, print a message and skip insertion
    if existing_cv:
        print("This CV already exists in the database.")
    else:
        structured_data['cv_hash'] = cv_hash # Add the hash to the structured data to store in the database
        structured_data['filename'] = filename # Store the filename in the structured data for reference
        cv_collection.insert_one(structured_data)# Insert the structured data into the MongoDB collection
        print("CV saved to the database.") # Print a confirmation message after successful insertion

# Function to extract text from PDFs using PyMuPDF
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf") # Open the PDF file from the binary stream (in-memory file object)
        return " ".join([page.get_text() for page in doc]) # Extract text from all pages and join them into a single string
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}") # Catch any error during the PDF reading or text extraction process
        raise # Re-raise the exception so it can be handled by the caller

# Function to extract text from DOCX files
def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file) # Open the DOCX file using the docx.Document class
        return "\n".join([para.text for para in doc.paragraphs]) # Extract text from each paragraph and join them with newline characters
    except Exception as e: 
        print(f"Error extracting text from DOCX: {str(e)}") # Catch any error during DOCX reading or extraction
        raise # Re-raise the exception so it can be handled by the caller

# Function to extract contact information
def extract_contact_info(cv_text):
    email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0.-]+\.[A-Z|a-z]{2,}\b', cv_text) # Use a regex pattern to find email addresses in the CV text
    phone = re.findall(r'\b\d{10,13}\b', cv_text)  # A basic pattern for phone numbers
    return {
        "email": email[0] if email else "Not Found", # Take the first matched email or "Not Found"
        "phone": phone[0] if phone else "Not Found"  # Take the first matched phone or "Not Found"
    }

# Function to extract education details simplified regex logic
def extract_education(cv_text):
    # List of keywords related to education that we want to search for in the CV text
    education_keywords = ["Bachelor", "Master", "PhD", "BSc", "MSc", "MBA", "BA", "BS", "MS", "University", "College", "Degree"]
    education_section = "" # Initialize an empty string to store extracted education details
    # Loop through each keyword in the list of education keywords
    for keyword in education_keywords: 
        # Search for the keyword followed by any text until a newline character
        match = re.search(rf'{keyword}.*\n', cv_text, re.IGNORECASE)
        # If a match is found, add the matched text to the education_section
        if match:
            education_section += match.group() + '\n' # Append the matched result with a newline
    # Return the collected education details, removing any extra whitespace, otherwise return "Not Found"
    return education_section.strip() if education_section else "Not Found" 

# Function to extract work experience 
def extract_experience(cv_text):
    # List of keywords related to work experience that we want to search for in the CV text
    experience_keywords = ["Experience", "Work History", "Employment", "Projects", "Responsibilities", "Duties"]
    experience_section = "" # Initialize an empty string to store extracted experience details
    # Loop through each keyword in the list of experience keywords
    for keyword in experience_keywords:
        # Search for the keyword followed by multiple lines of text
        match = re.search(rf'{keyword}.*\n(.*\n)+', cv_text, re.IGNORECASE)
        # If a match is found, add the matched text to the experience_section
        if match:  
            experience_section += match.group() + '\n' # Append the matched result with a newline
    # Return the collected experience details, removing any extra whitespace, otherwise return "Not Found"
    return experience_section.strip() if experience_section else "Not Found"

# Function to get skills from ESCO API
def get_skills_from_esco(job_title):
    esco_api_url = "https://ec.europa.eu/esco/api/search" # Define the base URL for the ESCO API search endpoint
    try:
        # Send a GET request to the ESCO API to search for the specified job title
        response = requests.get( 
            esco_api_url,
            params={
                'text': job_title, # Specify the job title to search for
                'type': 'occupation', # Filter the search to look for occupations only
                'limit': 1  # Get the best match
            }
        )
        if response.status_code == 200: # Checks if the request was successful 
            esco_data = response.json() # Converts the response to JSON format
            if esco_data and esco_data['results']: # Check if the response contains data and the 'results' field is populated
                occupation_uri = esco_data['results'][0]['uri'] # Extract the URI for the best-matching occupation

                # Fetch skills based on occupation URI
                skills_response = requests.get(f"https://ec.europa.eu/esco/api/resource/occupation/{occupation_uri}/skills")
                if skills_response.status_code == 200:
                    skills_data = skills_response.json() # Convert the skills response to JSON
                    return [skill['title'] for skill in skills_data] # Extract and return a list of skill titles from the JSON response
                else:
                    # Print an error message if the skills request fails
                    print(f"Error retrieving skills from ESCO: {skills_response.status_code}") 
                    return [] # Returns empty list if skills retrieval fails
            else:
                print("No matching occupation found") # Print an error message if no matching occupation is found
                return [] # Return an empty list if no match is found
        else:
            # Print an error message if the occupation search request fails
            print(f"Error retrieving occupation from ESCO: {response.status_code}")
            return [] # Return an empty list if the request fails
    except Exception as e:
        print(f"Error calling ESCO API: {str(e)}") # Catch any exceptions that occur during the API calls
        return [] # Return an empty list if an exception occurs
    
def extract_skills(cv_text):
    global extracted_skills_ESCO # Access the global list of dynamically fetched skills
    skills = [] # Initialize an empty list to store matched skills   
    # Predefined list of skills to search for in the CV text
    predefined_skills = [
# Programming Languages
    "python", "r", "sql", "java", "c++", "c#", "scala", "javascript", "typescript", "go", 
    "ruby", "php", "kotlin", "swift", "rust", "bash", "powershell", "matlab", "objective-c",
    "perl", "dart", "haskell", "lua", "f#", "julia", "groovy", "shell scripting", "smalltalk",

    # Machine Learning & AI
    "tensorflow", "pytorch", "scikit-learn", "keras", "huggingface", "openai", "chatgpt",
    "bert", "gpt-3", "llm", "transformers", "xgboost", "lightgbm", "catboost", "nltk", 
    "spacy", "deep learning", "cnn", "rnn", "gan", "vae", "autoencoders", "neural networks", 
    "natural language processing", "transfer learning", "fine-tuning", "reinforcement learning",
    "graph neural networks", "contrastive learning", "clustering", "classification", 
    "anomaly detection", "generative ai", "few-shot learning", "self-supervised learning",
    "semi-supervised learning", "zero-shot learning", "bayesian networks","data analysis", 
    "data preprocessing",

    # Data Science & Analytics
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "dash", "statsmodels", 
    "shap", "lime", "mlflow", "kubeflow", "feature engineering", "data cleaning", 
    "data wrangling", "time series analysis", "statistical modeling", "hypothesis testing", 
    "ab testing", "anova", "decision trees", "random forests", "support vector machines",
    "gradient boosting", "dimensionality reduction", "feature extraction", "data pipelines", 

    # Cloud Platforms & Services
    "aws", "azure", "gcp", "ibm cloud", "oracle cloud", "aws lambda", "aws ec2", "aws s3",
    "bigquery", "cloud run", "cloud functions", "vertex ai", "aws glue", "redshift", 
    "cloudwatch", "azure synapse", "azure databricks", "blob storage", "aws sagemaker", 
    "aws cloudformation", "terraform", "kubernetes", "eks", "aks", "gke", "helm", 
    "elastic beanstalk", "azure app service", "firebase", "cloud storage", "cloud security", 
    "cloud migration", "hybrid cloud", "multi-cloud", 

    # Web Development
    "html5", "css3", "sass", "less", "bootstrap", "tailwind", "javascript", "jquery", 
    "react", "angular", "vue.js", "next.js", "nuxt.js", "svelte", "astro", "django", 
    "flask", "fastapi", "express", "nestjs", "hapi.js", "graphql", "apollo", 
    "socket.io", "rest api", "soap", "grpc", 

    # Backend Development
    "node.js", "spring boot", "express.js", "django", "flask", "fastapi", "rust actix",
    "go fiber", "java servlet", "php laravel", "ruby on rails", "gin", "asp.net", 

    # Frontend Frameworks
    "react", "angular", "vue", "svelte", "solidjs", "preact", "next.js", "nuxt.js", 
    "tailwind", "bootstrap", "material ui", "chakra ui", 

    # Mobile Development
    "react native", "flutter", "kotlin", "swift", "xamarin", "phonegap", "capacitor", 
    "android studio", "xcode", 

    # DevOps & Automation
    "docker", "kubernetes", "terraform", "ansible", "puppet", "chef", "jenkins", "circleci", 
    "travis ci", "gitlab ci/cd", "github actions", "argo cd", "vault", "helm", "docker compose",
    "spinnaker", "prometheus", "grafana", "splunk", "datadog", 

    # Networking & Security
    "tcp/ip", "udp", "dns", "vpn", "firewall", "load balancing", "ssl/tls", "https", 
    "http/2", "http/3", "ipv4", "ipv6", "proxy", "nginx", "apache", "cisco ios", 
    "bgp", "ospf", "eigrp", "wireshark", "pfSense", "ipsec", "openvpn", "cloudflare", 

    # Cybersecurity
    "penetration testing", "ethical hacking", "metasploit", "nessus", "burp suite", 
    "red team", "blue team", "vulnerability scanning", "threat hunting", "malware analysis", 
    "siem", "soc", "snort", "cylance", "crowdstrike", "carbon black", "darktrace", 
    "ssl inspection", "sandboxing", "network segmentation", "threat intelligence", 
    "security orchestration", "ai-driven threat detection", "incident response", 
    "encryption protocols", "firewall configuration", "ids/ips", "risk assessment", 
    "patch management", "endpoint protection", "splunk", "qualys", "wireshark", "cisco", 
    "security auditing", "active directory", "group policy", 

    # Database Management
    "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle db", "ms sql server", 
    "couchbase", "neo4j", "elasticsearch", "dynamodb", "amazon aurora", "firestore", 
    "influxdb", "timescaledb", 

    # Testing & QA
    "selenium", "cypress", "playwright", "junit", "pytest", "testng", "mocha", 
    "jasmine", "karma", "appium", "loadrunner", "postman", "rest assured", 

    # Operating Systems
    "linux", "ubuntu", "debian", "red hat", "centos", "kali linux", "windows server", 
    "macos", "unix", "freebsd", "openbsd", 

    # Blockchain & Cryptography
    "ethereum", "solidity", "bitcoin", "hyperledger", "cardano", "defi", "zero knowledge proofs", 
    "cryptography", "openssl", "jwt", "sha256", "elliptic curve", 

    # Data Formats & Parsing
    "json", "xml", "csv", "parquet", "orc", "yaml", "protobuf", "avro", 

    # Search & Indexing
    "elasticsearch", "solr", "opensearch", "algolia", "redisearch", "vector search", 
    "k-nearest neighbors", 

    # Visualization & BI
    "tableau", "power bi", "looker", "metabase", "superset", "google data studio", 
    "dash", "plotly", "seaborn", "bokeh", 

    # Game Development
    "unity", "unreal engine", "godot", "game physics", "rendering", "shader programming", 

    # Mathematical & Statistical Skills
    "linear algebra", "probability", "statistics", "calculus", "bayesian inference", 
    "game theory", "graph theory", 

    # Additional Skills
    "pytorch lightning", "scalability", "high availability", "distributed systems", 
    "message queues", "event-driven architecture", "api gateway", "microservices", 
    "load balancing", "caching", "rate limiting", "feature flags"
]

    all_skills = list(set(predefined_skills + extracted_skills_ESCO))  # Remove duplicates
    cv_text = re.sub(r'\s+', ' ', cv_text).strip().lower() # Clean up text by removing extra spaces and normalizing cases    
    # Clean up each skill in the list of all skills
    for i in range(len(all_skills)):
        # Remove extra spaces and normalize case for consistent matching
        all_skills[i] = re.sub(r'\s+', ' ', all_skills[i]).strip().lower()

    # Use fuzzy matching to extract the best possible matches
    # `process.extract()` returns the top matches based on similarity score
    # `fuzz.token_set_ratio` allows matching even if the order of words is different
    potential_matches = process.extract(cv_text, all_skills, scorer=fuzz.token_set_ratio, limit=20)
    
    for match, score in potential_matches:
        # Lower threshold for multi-word terms to allow flexible matching
        if ' ' in match:
            threshold = 65  # Lower threshold for multi-word terms to account for small variations
        else:
            threshold = 70 # Higher threshold for single-word terms for stricter matching
        # Only consider matches that exceed the similarity threshold
        if score > threshold:
            # Length penalty – avoid partial or unrelated matches by limiting length difference
            # Allowing a buffer of 10 characters to account for reasonable length differences
            if len(match) <= len(cv_text) + 10:  
                # First context check:
                # Direct word boundary check to avoid false positives from partial words
                # Partial ratio allows for substring-based similarity checks
                if f" {match} " in f" {cv_text} " or fuzz.partial_ratio(match, cv_text) > 80:
                    skills.append(match)

                # Second context check:
                # Split the multi-word terms and try to match individual words using partial_ratio
                # Allows catching skills embedded in longer or complex sentences
                elif any(fuzz.partial_ratio(word.lower(), cv_text.lower()) > 80 for word in match.split()):
                    skills.append(match)

    return skills if skills else "Not Found" # Return matched skills if found; otherwise, return "Not Found"

# Function to organize CV data
def organize_cv_data(cv_text):
    return {
        "contact_info": extract_contact_info(cv_text), # Extract contact information (email and phone) from the CV text
        "education": extract_education(cv_text), # Extract educational background from the CV text
        "experience": extract_experience(cv_text), # Extract work experience details from the CV text
        "skills": extract_skills(cv_text), # Extract skills listed in the CV text
        "cv_text": cv_text  # Full CV text is also stored in case something is not picked up
    }

# Function to calculate weighted skill score
def calculate_skill_score(user_skills, game_skills):
    total_score = 0 # Initialize the total score to zero
    # Loop through each skill and its weight in the game_skills dictionary
    for skill, weight in game_skills.items():
        # Check if the current skill is present in the user's skill set
        if skill in user_skills:
            # If the skill is matched, add the weighted score (scaled to percentage) to the total score
            total_score += weight * 100  # Scale up the weights to percentages
    return total_score # Return the final calculated score after the loop finishes

# Home route
@app.route('/')
# home function that renders the main page
def home():
    return render_template('LSBUPhase1.html') #LSBUPhase1.html


#-------------------------------------------------------------------------------orginal Upload CV method
# Route to upload CV
@app.route('/upload_cv', methods=['POST']) # Defines a route '/upload_cv' that listens for POST requests
def upload_cv():
    uploaded_file = request.files['file'] # Retrieves the uploaded file from the form data under the key 'file'

    if uploaded_file:  # Checks if a file was uploaded
        try:
            # Save the file temporarily
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename) # Constructs the full file path using the upload folder and filename
            uploaded_file.save(file_path) # Saves the uploaded file to the constructed path
            file_extension = uploaded_file.filename.split('.')[-1].lower() # Extracts the file extension (e.g., pdf, docx) and converts to lowercase
            print(f"Received file: {uploaded_file.filename}")  # Debugging print
            if file_extension == 'pdf': # checks if the file is a PDF
                with open(file_path, 'rb') as pdf_file: # Open the PDF file in binary read mode
                    cv_text = extract_text_from_pdf(pdf_file) # Extract text from the PDF file using the extract_text_from_pdf() function
                print("PDF processed successfully") # Debugging print to confirm successful PDF processing
            elif file_extension == 'docx': # checks if the file is a DOCX
                with open(file_path, 'rb') as docx_file: # Open the DOCX file in binary read mode
                    cv_text = extract_text_from_docx(docx_file) # Extract text from the DOCX file using the extract_text_from_docx() function
                print("DOCX processed successfully") # Debugging print to confirm successful DOCX processing
            else:
                print("Unsupported file format") # Debugging print to indicate unsupported format
                return jsonify({'error': 'Unsupported file format'}), 400 # Return an error message with the status code 400 

            processed_cv_text = preprocess_text(cv_text) # Clean and preprocess the extracted CV text
            structured_data = organize_cv_data(processed_cv_text) # Organize the extracted data such as skills, contact info, and more

            print("Processed CV data:", structured_data)  # Debugging print

            pdf_url = url_for('view_pdf', filename=uploaded_file.filename) # Generate a URL to view the uploaded file
            print(f"PDF URL: {pdf_url}")  # Debugging print, shows the pdf URL

            return jsonify({ # Return the extracted data as a JSON response
                'skills': structured_data['skills'], # Include extracted skills
                'cv_text': cv_text, # Include unmodified CV text
                'pdf_url': pdf_url,  # Return the PDF URL
                'contact_info': structured_data['contact_info']  # Include contact information
            })
        except Exception as e: # Handle any errors that occur during processing
            print(f"Error processing the file: {str(e)}")  # Debugging print to display the error message
            return jsonify({'error': f'Error processing the file: {str(e)}'}), 500 # Return error message with the status code 500 
    else:
        print("No file uploaded")  # Debugging print, to help indicate file is not uploaded
        return jsonify({'error': 'No file uploaded'}), 400 # Return an error message with the status code 400


@app.route('/search_jobs', methods=['GET']) # Defines a Flask route at the endpoint '/search_jobs' to handle GET requests
def search_jobs():
    job_title = request.args.get('job_title') # Retrieves the 'job_title' parameter from the request URL
    esco_api_url = "https://ec.europa.eu/esco/api/search" # URL for the ESCO API endpoint
    
    try:
        # Call ESCO API to search for job titles
        response = requests.get( # Sends a GET request to the ESCO API
            esco_api_url, 
            params={ 
                'text': job_title, # Passes the job title as a query parameter to the API
                'type': 'occupation', # Specifies that the API should search for occupations
                'limit': 5  # Return top 5 matches
            }
        )
        
        # Debugging: Print the API response or status code
        if response.status_code == 200: # Checks if the API response is successful
            esco_data = response.json() # Parses the JSON response into a Python dictionary
            print("ESCO Job search response:", esco_data)  # Debugging print

            # Ensure we access the correct path to 'results'
            if '_embedded' in esco_data and 'results' in esco_data['_embedded']: # Checks if the 'results' key exists in the response
                jobs = [result['title'] for result in esco_data['_embedded']['results']] # Extracts job titles from the API response
                return jsonify({'jobs': jobs}) # Returns the list of job titles as a JSON response
            else:
                return jsonify({'error': 'No results found in ESCO API'}), 500  # Returns an error if no results are found
        else:
            print(f"Error in job search, status code: {response.status_code}")  # Debugging print shows the error status code
            return jsonify({'error': 'Error retrieving jobs from ESCO API'}), 500 # Returns an error message with status code 500
    except Exception as e:
        print(f"Error calling ESCO API: {str(e)}")  # Debugging print to display the exception details
        return jsonify({'error': str(e)}), 500 # Returns an error message with status code 500 if an exception occurs

@app.route('/get_job_skills', methods=['GET'])  # Defines a Flask route at the endpoint '/get_job_skills' to handle GET requests
def get_job_skills():
    global extracted_skills_ESCO  # Declare that we are using the global variable
    job_title = request.args.get('job_title') # Get the 'job_title' parameter from the request URL
    
    # Step 1: Search for the job/occupation using searchGet API
    search_url = f"https://ec.europa.eu/esco/api/search?text={job_title}&type=occupation&limit=5" # Format the search URL with the job title and set the limit to 5 results
    search_response = requests.get(search_url) # Send a GET request to the ESCO API search endpoint
    
    if search_response.status_code == 200: # Check if the search request was successful
        search_results = search_response.json() # Convert the JSON response to a Python dictionary
        if '_embedded' in search_results and 'results' in search_results['_embedded']: # Check if 'results' exists in the response
            occupation = search_results['_embedded']['results'][0] # Take the first occupation from the search results
            occupation_uri = occupation['uri'] # Extract the URI of the selected occupation
            print(f"Occupation URI: {occupation_uri}") # Output the occupation URI to the console
            skill_api_url = "https://ec.europa.eu/esco/api/resource/skill" # Define the ESCO skill API endpoint
            skill_response = requests.get(skill_api_url, params={'uri': occupation_uri}) # Send a GET request to the skill API using the occupation URI
            
            # Debugging print if needed
            print(f"Fetching skills from: {skill_api_url}?uri={occupation_uri}") # Output the full skill request URL
            print(f"Skills Response Status Code: {skill_response.status_code}") # Output the status code of the skill request
            
            if skill_response.status_code == 200: # Check if the skill request was successful
                # Extract and deduplicate titles
                titles = re.findall(r'"title":\s*"([^"]+)"', skill_response.content.decode('utf-8')) # Find all skill titles using regex
                extracted_skills_ESCO = list(set(titles))  # Remove duplicates and store as a list

                print("\nExtracted Titles:", extracted_skills_ESCO)  # Output deduplicated titles to the terminal

                skills_data = skill_response.json() # Convert the skill response to JSON
                return jsonify({
                    'skills_data': skills_data,  # Include the raw JSON if necessary
                    'extracted_titles': extracted_skills_ESCO   # Titles extracted for skill comparison
                })
            else:
                print(f"Error fetching skills: {skill_response.status_code}") # Output an error message if skill fetching fails
                return jsonify({'error': 'Error fetching job skills'}), 500 # Return an error response with status code 500
        else:
            return jsonify({'error': 'No occupations found'}), 404 # Return a 404 error if no occupations are found in the search results
    else:
        return jsonify({'error': 'Error fetching job suggestions'}), 500 # Return a 500 error if the search request fails

# Route to view PDF
@app.route('/view_pdf/<filename>') # Defines a Flask route at the endpoint '/view_pdf/<filename>' to serve PDF files
def view_pdf(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename) # file path using the upload folder and the provided filename
    if os.path.exists(file_path): # Checks if the file exists at the specified path
        try:
            return send_file(file_path) # If the file exists, serve it to the client using Flask's send_file() function
        except Exception as e: # Catch any exception that occurs while serving the file
            print(f"Error serving PDF file: {str(e)}")  # Debugging print to display the exception details
            return jsonify({'error': str(e)}) # Return a JSON response with the error message
    else:
        print(f"File not found: {filename}")  # Debugging print to indicate that the file was not found
        return jsonify({'error': f'File {filename} not found'}), 404 # Return a 404 error if the file doesnt  exist

# Route to test the MongoDB connection
@app.route('/test_mongodb')  # Defines a Flask route at the endpoint '/test_mongodb'
def test_mongodb(): # Function to handle the request to the '/test_mongodb' route
    try:
        cv_count = cv_collection.count_documents({}) # Count the number of documents in the 'cv_collection' collection
        return f'MongoDB connected. {cv_count} CVs found in the database.' # Return a success message with the number of CVs found
    except Exception as e:
        return f'Error connecting to MongoDB: {str(e)}' # If an error occurs, return an error message with the exception details

# Route to process skills and save them
@app.route('/process_skills', methods=['POST']) # Defines a Flask route at the endpoint '/process_skills' to handle POST requests
def process_skills():
    skills = request.form.getlist('skills') # Extracts the list of skills from the form data
    cv_text = request.form['cv_text'] # Extracts the raw CV text from the form data
    years_of_experience = request.form['experience'] # Extracts the years of experience from the form data

    # Extracts contact information (email and phone) from the form data and stores it in a dictionary
    contact_info = {
        'email': request.form['email'], # Extracts the email address
        'phone': request.form['phone'] # Extracts the phone number
    }
    structured_data = organize_cv_data(cv_text) # Organizes the extracted data into a structured format using the organize_cv_data() function
    structured_data['skills'] = skills # Adds the extracted skills to the structured data
    structured_data['years_of_experience'] = years_of_experience # Adds the years of experience to the structured data
    structured_data['contact_info'] = contact_info # Adds the contact info email and phone to the structured data

    # Saves the structured data to MongoDB using the save_cv_to_mongodb() function
    save_cv_to_mongodb("user_uploaded_cv", structured_data)

    # Returns a JSON response confirming that the data was saved successfully
    return jsonify({'message': 'Skills, experience, and contact information saved successfully'})

# Route to display games and calculate skill match
@app.route('/games')
def games():
    games_data = [ # Define the list of available games with their required skills, image, and required score to unlock
        {
            "name": "Website Quiz", # title of the game/quiz/survey
            "skills_required": {"C++": 0.5, "Java": 0.5}, # Skills required and their weights for the game
            "image": "https://www.stx.ox.ac.uk/sites/default/files/stx/images/article/depositphotos_41197145-stock-photo-quiz.jpg", # URL for the game's image
            "required_score": 75 # Minimum score required to unlock the game
        },
        {
            "name": "Ai Survey", # title of the game/quiz/survey
            "skills_required": {"machine learning": 0.25, "python": 0.50, "tensorflow": 0.25}, # Skills required and their weights for the game
            "image": "https://www.questionpro.com/blog/wp-content/uploads/2024/02/AI-Survey.jpg",# URL for the game's image
            "required_score": 75 # Minimum score required to unlock the game
        },
        {
            "name": "Survey", # title of the game/quiz/survey
            "skills_required": {"React": 0.5, "JavaScript": 0.5}, # Skills required and their weights for the game
            "image": "https://img.freepik.com/free-vector/online-survey-tablet_3446-296.jpg", # URL for the game's image
            "required_score": 75 # Minimum score required to unlock the game
        },
        {
            "name": "Research Survey", # title of the game/quiz/survey
            "skills_required": {"docker": 0.5, "aws": 0.5}, # Skills required and their weights for the game
            "image": "https://www.aimtechnologies.co/wp-content/uploads/2024/02/Types-of-Survey-Research.jpg", # URL for the game's image
            "required_score": 75 # Minimum score required to unlock the game
        }
    ]
    user_cv = cv_collection.find_one(sort=[("_id", -1)])  # Get the latest uploaded CV    
    # Extract the list of user skills from the CV and convert them to lowercase for comparison
    user_skills = [skill.lower() for skill in user_cv.get("skills", [])] if user_cv else []
    response_data = [] # Initialize an empty list to store game results
    # Loop through each game and calculate the user's eligibility based on skills
    for game in games_data: 
        game_skills = list(game['skills_required'].keys()) # Get the list of required skills for the game

        # Calculate skills the user has and is missing
        matched_skills = [skill for skill in game_skills if skill in user_skills] # Identify matched skills
        missing_skills = [skill for skill in game_skills if skill not in user_skills] # Identify missing skills

        # Calculate user's score based on the matched skills
        user_score = calculate_skill_score(matched_skills, game['skills_required'])
        can_play = user_score >= game['required_score'] # Determine if the user can unlock the game based on their score
        # Set the game status based on the user's score
        game_status = "You can Take this Survey/Quiz!" if can_play else f"You need {game['required_score'] - user_score}% more to unlock this Survey/Quiz."

        # Append the calculated game info to the response list
        response_data.append({
            "name": game['name'], # Name of the game
            "image": game['image'], # URL for the game's image
            "status": game_status, # Display message based on the user's score
            "user_score": user_score, # User's calculated score
            "required_score": game['required_score'], # Score needed to unlock the game
            "matched_skills": matched_skills, # List of matched skills
            "missing_skills": missing_skills # List of missing skills
        })
    return jsonify({"games": response_data, "user_skills": user_skills}) # Return the game data and user skills as a JSON response

# Route to get game details
@app.route('/game/<game_name>')
def game_detail(game_name):
    game_details = { # Define the detailed information for each game
        "Website Quiz": {
            "skills_required": ["C++", "Java"], # Skills required for this game
            "description": "A web development game that involves coding in C++ and Java.", # Description of the game
            "image": "https://www.stx.ox.ac.uk/sites/default/files/stx/images/article/depositphotos_41197145-stock-photo-quiz.jpg" # URL for the game's image
        },
        "Ai Survey": {
            "skills_required": ["machine learning", "python", "tensorflow"], # Skills required for this game
            "description": "An AI-based survey tool requiring Python, TensorFlow, and machine learning expertise.", # Description of the game
            "image": "https://devskrol.com/wp-content/uploads/2021/10/PythonQ1S2-1.jpg" # URL for the game's image
        },
        "Survey": {
            "skills_required": ["React", "JavaScript"], # Skills required for this game
            "description": "A card-based game where you need React and JavaScript skills to develop the front end.", # Description of the game
            "image": "https://img.freepik.com/free-vector/online-survey-tablet_3446-296.jpg" # URL for the game's image
        },
        "Research Survey": {
            "skills_required": ["docker", "aws"], # Skills required for this game
            "description": "A space simulation game where Docker and AWS are required to handle cloud operations.", # Description of the game
            "image": "https://www.aimtechnologies.co/wp-content/uploads/2024/02/Types-of-Survey-Research.jpg" # URL for the game's image
        }
    }
    # Check if the requested game exists in the dictionary
    if game_name in game_details:
        game_info = game_details[game_name] # Fetch the game details
        game_info["matched_skills"] = []  # Add matched skills here if available
        game_info["missing_skills"] = []  # Add missing skills here if available
        return jsonify(game_info) # Return the game details as a JSON response
    else:
        return jsonify({"error": "Game not found"}), 404 # Return an error message if the game is not found

# Email-related functions for reminders
def get_user_emails():
    user_emails = [] # Initialize an empty list to store user emails
    for cv in cv_collection.find({}, {'contact_info.email': 1}): # Query MongoDB to get the 'email' field from the 'contact_info' document
        email = cv.get('contact_info', {}).get('email', None) # Get the email field; return None if not found
        if email:
            user_emails.append(email) # If email exists, add it to the list
    return user_emails # Return the list of emails

def send_email(recipient_email, subject, message):
    smtp_server = "smtp.gmail.com" # Define the SMTP server for Gmail
    smtp_port = 587 # Define the SMTP port for Gmail
    smtp_user = "email@gmail.com"  # Sender's email address
    smtp_password = "Password"  # Sender's email password App Password

    msg = MIMEMultipart()
    msg['From'] = smtp_user # Set the sender's email address
    msg['To'] = recipient_email # Set the recipient's email address
    msg['Subject'] = subject # Set the subject of the email
    msg.attach(MIMEText(message, 'plain')) # Attach the message body as plain text

    try:
        server = smtplib.SMTP(smtp_server, smtp_port) # Connect to the SMTP server
        server.starttls() # Start TLS encryption for a secure connection
        server.login(smtp_user, smtp_password) # Log into the SMTP server using credentials
        server.send_message(msg) # Send the email message
        server.quit() # Close the connection to the server
        print(f"Email sent to {recipient_email}") # Print a success message
    except Exception as e:
        print(f"Failed to send email to {recipient_email}: {e}") # Print an error message if sending fails

# Function to create the reminder message
def create_email_message():
    link_to_games = "http://127.0.0.1:5001/games" # URL link to the games page
    message = f"""
    Hello,

    This is a reminder that new quizs/surveys and more are out click on the link below to view them:

    {link_to_games}

    Best regards,
    Your Game Platform Team
    """
    return message # Return the formatted message

# Function to send reminders to all users
def send_reminders():
    user_emails = get_user_emails() # Get the list of user emails from MongoDB
    message = create_email_message() # Create the email message
    subject = "New Games Unlocked - Check them out!"  # Define the email subject

    for email in user_emails: # Loop through all user emails
        send_email(email, subject, message) # Send the reminder email to each user

# Schedule email reminders every Wednesday 3:00pm
schedule.every().wednesday.at("15:00").do(send_reminders) # Use the schedule library to set up automatic reminders

print("Scheduler is running...") # for debugging

# Run the scheduler
def run_scheduler():
    while True:
        schedule.run_pending() # Check if any scheduled tasks are ready to run
        time.sleep(60) # Wait for 60 seconds before checking again

# Start the scheduler in a separate thread if needed, or directly in __main__
if __name__ == "__main__":
    import threading # Import threading to run scheduler in parallel
    scheduler_thread = threading.Thread(target=run_scheduler) # Create a thread for the scheduler
    scheduler_thread.start() # Start the scheduler thread
    app.run(debug=True, port=5001) # Start the Flask app on port 5001 in debug mode




#peterpan10133@gmail.com
#avaoafezicbxljjj








