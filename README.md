# 🧠 AI-Driven CV Analysis Model

A 3-Phase intelligent system that analyzes CVs, extracts skills, and matches them to job descriptions or gamified tasks using AI and NLP.

---

## 📌 Overview

This project was built as part of my BEng final year project at LSBU and also contributes to the Play2Earn.ai platform. It includes CV skill extraction, job description matching, and AI-based semantic comparison between CVs and JDs.

---

## 🚀 Project Objectives

- Phase 1: Upload a CV and match skills to games/quizzes/surveys
- Phase 2: Upload a job description and find the best matching CVs from a database that you created in Phase 1 
- Phase 3: Upload multiple CVs and JDs and compute AI similarity scores between them and see best fits for each pair

---

## 🧠 Technologies Used

- Python (Flask)
- NLP (fuzzywuzzy, regex, stopwords)
- SentenceTransformers (MPNet)
- MongoDB
- ESCO API
- HTML/CSS/JS (with jQuery for AJAX)

---

## 🖥 How to Run

### Step 1: MongoDB Setup #########

This project uses MongoDB to store parsed CV data across phases.

#### Option 1: Run MongoDB Locally

1. Install MongoDB Community Edition:  
   https://www.mongodb.com/try/download/community 

2. Start the MongoDB service:

```bash
# macOS/Linux
brew services start mongodb-community

# Windows (use MongoDB Compass or Services.msc)

### Step 2: Dependencies intallation #########
pip install -r requirements.txt

### Step 3: Project Folder Structure #########
Use VS Code for multiple language usage (python, Html, javascript) or any other compatible program and structure folder like this.

AI-CV-Analysis-Model/
├── Phase1LSBU.py
├── Phase2LSBU.py
├── Phase3LSBU.py
├── templates/
│   ├── LSBUPhase1.html
│   ├── LSBUPhase2.html
│   └── LSBUPhase3.html













