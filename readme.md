# NLTI CLAT Assistant

## Overview
NLTI CLAT Assistant is a Streamlit-based web application designed to help CLAT (Common Law Admission Test) aspirants in their preparation journey. The app provides personalized mentor recommendations and answers to common questions about CLAT exams through an interactive interface.

## Features

### 1. Mentor Recommendation System
- **Personalized Matching**: Find mentors who match your learning style, subject preferences, and target law school
- **Comprehensive Profiles**: View detailed mentor profiles including expertise, teaching style, and availability
- **Smart Matching Algorithm**: Uses cosine similarity to match students with the most suitable mentors

### 2. CLAT Query Assistant
- **Interactive Chatbot**: Get instant answers to questions about CLAT exams
- **Knowledge Base**: Comprehensive information about syllabus, exam pattern, preparation strategies, and more
- **Natural Language Processing**: Uses text preprocessing and keyword matching for relevant responses


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Karthic-Elangovan/NLTI-CLAT-Assistant.git
cd NLTI-CLAT-Assistant
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

5. Access the app in your web browser at `http://localhost:8501`

## Usage

### Mentor Recommendation
1. Navigate to the "Mentor Recommendation" tab
2. Fill out your preferences, including:
   - Primary and secondary subjects of interest
   - Target law school
   - Preferred learning style
   - Current preparation level
   - Availability preferences
3. Provide additional information such as target exam date and weekly study hours
4. Click "Find My Mentors" to receive personalized recommendations

### CLAT Assistant
1. Navigate to the "CLAT Assistant" tab
2. Type your question about CLAT in the chat input
3. Receive instant answers from the knowledge base
4. Refer to the "Sample questions" section for query ideas

## Technical Details

### Mentor Recommendation System
- Uses One-Hot Encoding to transform categorical data
- Implements cosine similarity for matching user preferences with mentor profiles
- Ranks mentors based on match percentage

### CLAT Query Assistant
- Preprocesses text using NLTK for tokenization and stopword removal
- Uses keyword matching and context scoring to find relevant responses
- Includes error handling for robust performance

### Data
- Currently uses mock data for demonstration
- Can be extended to use real mentor profiles and expanded knowledge base

## Future Enhancements
- User authentication system
- Scheduling system for mentor sessions
- Expanded knowledge base with more topics
- Improved NLP capabilities for the assistant
- Integration with actual mentor databases
- User feedback collection and recommendation refinement

## Contributing
Contributions to improve NLTI CLAT Assistant are welcome! Feel free to submit pull requests or open issues to suggest improvements.