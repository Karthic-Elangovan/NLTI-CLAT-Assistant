import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import json

# Improved NLTK resource download function
def download_nltk_resources():
    # Only include resources that actually exist in NLTK
    resources = [
        'punkt',
        'stopwords'
    ]
    
    for resource in resources:
        try:
            # More straightforward approach to checking and downloading
            nltk.data.find(f"{'' if resource in ['punkt', 'stopwords'] else 'tokenizers/'}{resource}")
            print(f"Resource '{resource}' is already downloaded.")
        except LookupError:
            print(f"Downloading resource '{resource}'...")
            nltk.download(resource, quiet=True)
            print(f"Resource '{resource}' downloaded successfully.")

# Download required NLTK resources
download_nltk_resources()

# Set page configuration
st.set_page_config(
    page_title="NLTI CLAT Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("NLTI CLAT Assistant")
st.markdown("""
    This app helps CLAT aspirants by providing:
    1. **Personalized Mentor Recommendations** - Find mentors who match your learning style and goals
    2. **CLAT Query Assistant** - Get answers to common questions about CLAT exams
""")

# Create tabs for the two functionalities
tab1, tab2 = st.tabs(["Mentor Recommendation", "CLAT Assistant"])

# ----- MOCK DATA FOR DEMONSTRATION -----

# Generate mock mentor data
@st.cache_data
def load_mentor_data():
    # Create mock mentor data
    mentors_data = {
        'mentor_id': range(1, 21),
        'name': [
            'Arjun Sharma', 'Priya Patel', 'Rajiv Malhotra', 'Kavita Singh', 'Aditya Kumar',
            'Neha Gupta', 'Vikram Bose', 'Meera Chadha', 'Siddharth Jain', 'Divya Reddy',
            'Rohan Mehta', 'Ananya Roy', 'Karan Agarwal', 'Shreya Verma', 'Nikhil Chopra',
            'Tanya Bajaj', 'Amit Singhania', 'Sneha Kapoor', 'Varun Desai', 'Pooja Khanna'
        ],
        'strong_subjects': [
            'Constitutional Law', 'Legal Reasoning', 'Legal Aptitude', 'English', 'Quantitative Techniques',
            'Logical Reasoning', 'Current Affairs', 'English', 'Constitutional Law', 'Legal Reasoning',
            'Quantitative Techniques', 'Logical Reasoning', 'Current Affairs', 'Legal Aptitude', 'English',
            'Constitutional Law', 'Legal Reasoning', 'Logical Reasoning', 'Current Affairs', 'Legal Aptitude'
        ],
        'secondary_subjects': [
            'Legal Reasoning', 'Constitutional Law', 'English', 'Legal Reasoning', 'Legal Aptitude',
            'English', 'Logical Reasoning', 'Current Affairs', 'Legal Aptitude', 'English',
            'Constitutional Law', 'Legal Reasoning', 'English', 'Constitutional Law', 'Legal Reasoning',
            'Current Affairs', 'English', 'Constitutional Law', 'Legal Reasoning', 'English'
        ],
        'alma_mater': [
            'NLSIU Bangalore', 'NALSAR Hyderabad', 'NLIU Bhopal', 'WBNUJS Kolkata', 'NLU Delhi',
            'NLSIU Bangalore', 'NALSAR Hyderabad', 'NLIU Bhopal', 'WBNUJS Kolkata', 'NLU Delhi',
            'NLSIU Bangalore', 'NALSAR Hyderabad', 'NLIU Bhopal', 'WBNUJS Kolkata', 'NLU Delhi',
            'NLSIU Bangalore', 'NALSAR Hyderabad', 'NLIU Bhopal', 'WBNUJS Kolkata', 'NLU Delhi'
        ],
        'teaching_style': [
            'Interactive', 'Conceptual', 'Problem-based', 'Visual', 'Practice-oriented',
            'Interactive', 'Conceptual', 'Problem-based', 'Visual', 'Practice-oriented',
            'Interactive', 'Conceptual', 'Problem-based', 'Visual', 'Practice-oriented',
            'Interactive', 'Conceptual', 'Problem-based', 'Visual', 'Practice-oriented'
        ],
        'years_experience': [
            3, 5, 2, 4, 6, 2, 3, 5, 4, 2, 6, 3, 4, 2, 5, 3, 4, 2, 5, 3
        ],
        'clat_rank': [
            12, 5, 28, 15, 3, 21, 8, 17, 9, 31, 7, 22, 13, 25, 11, 19, 6, 24, 14, 27
        ],
        'bio': [
            'CLAT topper specialized in Constitutional Law with innovative teaching methods.',
            'Expert in Legal Reasoning with a structured approach to problem-solving.',
            'Focuses on legal aptitude with real-world case applications.',
            'English specialist with emphasis on critical reading techniques.',
            'Quant expert who makes numbers accessible for humanities students.',
            'Logical reasoning coach who breaks down complex problems.',
            'Current affairs specialist with legal perspective on news.',
            'English and comprehension expert focusing on exam techniques.',
            'Constitutional law expert with moot court experience.',
            'Legal reasoning mentor with previous teaching experience.',
            'Quant specialist who creates simplified frameworks.',
            'Reasoning expert with focus on pattern recognition.',
            'Current affairs guru with daily news analysis for CLAT.',
            'Legal aptitude coach with specialized material.',
            'English language specialist focusing on vocabulary building.',
            'Constitutional interpretation expert with judicial insights.',
            'Legal reasoning mentor focused on critical thinking.',
            'Logic and reasoning specialist with visual learning approach.',
            'Current affairs analyst with focus on legal implications.',
            'Legal principles expert with simplified learning methods.'
        ],
        'rating': [
            4.8, 4.9, 4.5, 4.7, 4.9, 4.6, 4.8, 4.7, 4.9, 4.5, 4.8, 4.6, 4.7, 4.5, 4.8, 4.7, 4.9, 4.6, 4.7, 4.5
        ],
        'availability': [
            'Weekends', 'Weekdays', 'Evenings', 'Mornings', 'Flexible',
            'Weekends', 'Weekdays', 'Evenings', 'Mornings', 'Flexible',
            'Weekends', 'Weekdays', 'Evenings', 'Mornings', 'Flexible',
            'Weekends', 'Weekdays', 'Evenings', 'Mornings', 'Flexible'
        ]
    }
    return pd.DataFrame(mentors_data)

# CLAT Knowledge Base
@st.cache_data
def load_clat_knowledge_base():
    # Sample knowledge base for CLAT-related queries
    knowledge_base = {
        "syllabus": {
            "keywords": ["syllabus", "course", "subjects", "topics", "curriculum"],
            "context": "CLAT 2025",
            "response": """The CLAT 2025 syllabus covers five main sections:
            
1. English Language (20% of the paper): Comprehension passages and grammar, vocabulary, etc.
2. Current Affairs & General Knowledge (25% of the paper): Including static GK and current affairs
3. Legal Reasoning (25% of the paper): Analyzing legal situations and applying principles
4. Logical Reasoning (20% of the paper): Logical arguments, facts, principles, etc.
5. Quantitative Techniques (10% of the paper): Mathematical concepts, data interpretation, etc.
            
The exam is conducted for 2 hours with 150 multiple-choice questions."""
        },
        "exam_pattern": {
            "keywords": ["pattern", "format", "structure", "marking", "negative marking", "questions", "duration", "time"],
            "context": "",
            "response": """CLAT exam pattern:
- Total questions: 150 multiple-choice questions
- Duration: 2 hours (120 minutes)
- Marking scheme: 1 mark for each correct answer
- Negative marking: 0.25 marks deducted for each wrong answer
- Mode of exam: Computer-based test
- Medium: English only"""
        },
        "english_section": {
            "keywords": ["english", "comprehension", "grammar", "vocabulary", "language"],
            "context": "questions",
            "response": """The English section in CLAT typically consists of 30-35 questions (20% of the paper), including:
- Reading comprehension passages
- Grammar questions
- Vocabulary-based questions
- Para jumbles and sentence completion
This section tests your reading and comprehension abilities, understanding of grammar rules, and vocabulary."""
        },
        "legal_reasoning": {
            "keywords": ["legal", "reasoning", "principles", "cases", "judgments"],
            "context": "",
            "response": """The Legal Reasoning section (25% of the paper) tests your ability to:
- Identify and apply legal principles to factual situations
- Analyze legal problems and come to a conclusion
- Understand legal concepts
You don't need prior legal knowledge as principles are provided in the passages."""
        },
        "logical_reasoning": {
            "keywords": ["logical", "reasoning", "arguments", "logic", "critical thinking"],
            "context": "",
            "response": """The Logical Reasoning section (20% of the paper) assesses:
- Ability to identify patterns, logical links, and rectify illogical arguments
- Deductive and inductive reasoning
- Critical thinking and analytical skills
It includes questions on analogies, syllogisms, logical sequences, and arguments."""
        },
        "quantitative_techniques": {
            "keywords": ["quant", "quantitative", "math", "mathematics", "numerical", "calculation"],
            "context": "",
            "response": """The Quantitative Techniques section (10% of the paper) covers:
- Basic mathematical concepts (class 10th level)
- Elementary algebra
- Data interpretation (graphs, charts)
- Simple arithmetic
It's designed to test basic mathematical aptitude rather than advanced skills."""
        },
        "current_affairs": {
            "keywords": ["current", "affairs", "gk", "general knowledge", "news"],
            "context": "",
            "response": """The Current Affairs & GK section (25% of the paper) covers:
- Important national and international events
- Legal news and developments
- Key appointments and awards
- Static GK (history, geography, polity, etc.)
Focus on events from the last 6-12 months before the exam."""
        },
        "cutoff": {
            "keywords": ["cutoff", "cut-off", "cutoffs", "cut-offs", "cut off", "score", "marks", "minimum"],
            "context": "NLSIU Bangalore",
            "response": """For NLSIU Bangalore (National Law School of India University), which is generally considered the top law school in India:
            
The cut-off for general category for the last year was approximately 110-115 out of 150 marks.
For reserved categories, the cut-offs were lower:
- SC/ST categories: 80-90 marks
- OBC: 95-105 marks

Please note that cut-offs vary year to year based on difficulty level of the paper and the performance of students."""
        },
        "preparation": {
            "keywords": ["prepare", "preparation", "strategy", "study", "tips", "advice"],
            "context": "",
            "response": """Effective CLAT preparation strategy:
1. Understand the syllabus thoroughly
2. Create a study schedule with dedicated time for each section
3. Read newspapers daily for current affairs
4. Practice reading comprehension to improve speed
5. Solve previous years' question papers
6. Take regular mock tests to build exam temperament
7. Focus on accuracy first, then speed
8. Maintain a current affairs diary
9. Join a coaching program if possible
10. Revise regularly and identify weak areas"""
        },
        "important_dates": {
            "keywords": ["dates", "deadline", "schedule", "calendar", "registration", "application", "when"],
            "context": "",
            "response": """Important dates for CLAT 2025:
- Application start date: August-September 2024 (tentative)
- Last date for application: Usually November 2024
- Admit card release: About 2 weeks before the exam
- Exam date: Typically in May 2025
- Result declaration: Usually within 2-3 weeks after the exam

Please check the official CLAT website for the most accurate and updated information on dates."""
        },
        "eligibility": {
            "keywords": ["eligibility", "eligible", "criteria", "qualification", "qualify", "requirement"],
            "context": "",
            "response": """Eligibility criteria for CLAT:
1. For UG programs (B.A. LL.B):
   - Minimum 45% marks in 10+2 or equivalent (40% for SC/ST categories)
   - No age limit (as per Supreme Court ruling)

2. For PG programs (LL.M):
   - LL.B degree or equivalent with minimum 55% marks (50% for SC/ST categories)

The qualifying exam (10+2) should be from a recognized board."""
        },
        "top_colleges": {
            "keywords": ["college", "colleges", "university", "universities", "institutions", "schools", "best", "top"],
            "context": "",
            "response": """Top law colleges accepting CLAT scores:
1. National Law School of India University (NLSIU), Bangalore
2. National Academy of Legal Studies and Research (NALSAR), Hyderabad
3. National Law Institute University (NLIU), Bhopal
4. West Bengal National University of Juridical Sciences (WBNUJS), Kolkata
5. National Law University (NLU), Delhi
6. National Law University (NLU), Jodhpur
7. Hidayatullah National Law University (HNLU), Raipur
8. Gujarat National Law University (GNLU), Gandhinagar
9. Dr. Ram Manohar Lohiya National Law University (RMLNLU), Lucknow
10. Rajiv Gandhi National University of Law (RGNUL), Patiala

These rankings may vary slightly year to year."""
        },
        "books": {
            "keywords": ["book", "books", "study material", "resources", "read", "guide", "material"],
            "context": "",
            "response": """Recommended books for CLAT preparation:

1. English:
   - Word Power Made Easy by Norman Lewis
   - High School Grammar and Composition by Wren & Martin

2. Legal Reasoning:
   - Legal Awareness and Legal Reasoning by A.P. Bhardwaj
   - Universal's Legal Reasoning for CLAT & LL.B. Entrance Examinations

3. Logical Reasoning:
   - A Modern Approach to Logical Reasoning by R.S. Aggarwal
   - Analytical Reasoning by M.K. Pandey

4. GK & Current Affairs:
   - Manorama Yearbook
   - Competition Success Review
   - Monthly magazines like Pratiyogita Darpan

5. Quantitative Techniques:
   - Quantitative Aptitude for Competitive Examinations by R.S. Aggarwal
   - NCERT Mathematics (Class 8-10)

6. General CLAT guides:
   - Universal's Guide to CLAT & LL.B. Entrance Examination
   - Pearson Guide to CLAT"""
        },
        "fees": {
            "keywords": ["fee", "fees", "cost", "expense", "financial", "tuition", "payment"],
            "context": "",
            "response": """CLAT application fee:
- General/OBC/PWD categories: ₹4,000
- SC/ST/BPL categories: ₹3,500

College fees at top NLUs (approximate annual fees):
1. NLSIU Bangalore: ₹2.3-2.8 lakhs per annum
2. NALSAR Hyderabad: ₹2.2-2.7 lakhs per annum
3. NLIU Bhopal: ₹1.8-2.2 lakhs per annum
4. WBNUJS Kolkata: ₹2.0-2.5 lakhs per annum
5. NLU Delhi: ₹2.1-2.6 lakhs per annum

These fees are approximate and may change. Most NLUs also offer scholarship programs for meritorious students and those from economically weaker sections."""
        },
        "difficulty": {
            "keywords": ["difficult", "difficulty", "tough", "easy", "harder", "easier", "hardest"],
            "context": "",
            "response": """The difficulty level of CLAT varies from year to year:

In recent years, CLAT has shifted towards a more comprehension-based pattern, making it moderately difficult. The focus is now on testing analytical skills rather than rote learning.

Section-wise difficulty (typically):
- English: Moderate to difficult (depends on passage complexity)
- Current Affairs: Moderate (requires regular reading)
- Legal Reasoning: Moderate to difficult (requires analytical thinking)
- Logical Reasoning: Moderate (requires practice)
- Quantitative Techniques: Easy to moderate (basic math skills)

The overall difficulty is managed to ensure appropriate differentiation among candidates. With proper preparation of 6-12 months, most students can achieve a good score."""
        }
    }
    return knowledge_base

# -- MENTOR RECOMMENDATION SYSTEM --

# Function to preprocess mentor data for recommendation
def preprocess_mentor_data(mentors_df):
    # Create features for matching
    features = mentors_df[['strong_subjects', 'secondary_subjects', 'alma_mater', 'teaching_style']]
    
    # One-hot encode categorical features
    # Fixed: Changed sparse=False to sparse_output=False for newer scikit-learn versions
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(features)
    
    # Return the encoder and encoded features
    return encoder, encoded_features, features.columns

# Function to get mentor recommendations
def get_mentor_recommendations(user_preferences, mentors_df, encoder, encoded_mentors, feature_names):
    # Extract user preferences
    user_data = pd.DataFrame({
        'strong_subjects': [user_preferences['preferred_subject']],
        'secondary_subjects': [user_preferences['secondary_subject']],
        'alma_mater': [user_preferences['target_college']],
        'teaching_style': [user_preferences['learning_style']]
    })
    
    # Encode user preferences
    encoded_user = encoder.transform(user_data)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(encoded_user, encoded_mentors)[0]
    
    # Get top 3 recommendations
    top_indices = similarities.argsort()[-3:][::-1]
    recommended_mentors = mentors_df.iloc[top_indices].copy()
    
    # Add similarity score as match percentage
    recommended_mentors['match_percentage'] = similarities[top_indices] * 100
    
    return recommended_mentors

# -- CHATBOT FOR CLAT QUERIES --

# Modified text preprocessing function with better error handling
def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Try to tokenize with NLTK, fall back to basic split if it fails
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            print(f"Tokenization error, falling back to basic split: {str(e)}")
            tokens = text.split()
        
        # Try to remove stopwords if available
        try:
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [word for word in tokens if word not in stop_words]
            return filtered_tokens
        except Exception as e:
            print(f"Stopwords removal error, returning tokens: {str(e)}")
            return tokens
            
    except Exception as e:
        print(f"Error in text processing: {str(e)}")
        # Return simple word split as fallback
        return text.lower().split()

# Function to find the most relevant response from knowledge base
def get_response(query, knowledge_base):
    try:
        query_tokens = preprocess_text(query)
        
        best_match = None
        highest_score = 0
        
        for topic, data in knowledge_base.items():
            # Check for keywords
            keyword_match = sum(1 for token in query_tokens if token in data['keywords'])
            
            # Check for context if present
            context_score = 0
            if data['context'] and data['context'].lower() in query.lower():
                context_score = 2
            
            # Calculate total score
            total_score = keyword_match + context_score
            
            if total_score > highest_score:
                highest_score = total_score
                best_match = data['response']
        
        # If no good match found
        if highest_score < 1 or best_match is None:
            return """I'm sorry, I don't have specific information about that query. 
                    Please try asking about CLAT syllabus, exam pattern, specific subjects, 
                    or other exam-related information. You can also try rephrasing your question."""
        
        return best_match
    except Exception as e:
        # Graceful error handling
        return f"I apologize, but I encountered an error processing your query. Please try rephrasing or ask another question."

# -- STREAMLIT UI --

# Load data
mentors_df = load_mentor_data()
knowledge_base = load_clat_knowledge_base()

# Preprocess mentor data for recommendation
encoder, encoded_mentors, feature_names = preprocess_mentor_data(mentors_df)

# Mentor Recommendation UI
with tab1:
    st.header("Find Your Perfect CLAT Mentor")
    st.write("Let us match you with mentors who best fit your preferences and learning style.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Preferences")
        preferred_subject = st.selectbox(
            "Which subject do you want to focus on?",
            options=['Constitutional Law', 'Legal Reasoning', 'Legal Aptitude', 'English', 
                     'Quantitative Techniques', 'Logical Reasoning', 'Current Affairs']
        )
        
        secondary_subject = st.selectbox(
            "Secondary subject interest",
            options=['Constitutional Law', 'Legal Reasoning', 'Legal Aptitude', 'English', 
                     'Quantitative Techniques', 'Logical Reasoning', 'Current Affairs']
        )
        
        target_college = st.selectbox(
            "Target law school",
            options=['NLSIU Bangalore', 'NALSAR Hyderabad', 'NLIU Bhopal', 'WBNUJS Kolkata', 'NLU Delhi']
        )
        
        learning_style = st.selectbox(
            "Your preferred learning style",
            options=['Interactive', 'Conceptual', 'Problem-based', 'Visual', 'Practice-oriented']
        )
        
        current_level = st.select_slider(
            "Your current preparation level",
            options=['Beginner', 'Intermediate', 'Advanced']
        )
        
        availability_pref = st.multiselect(
            "Your availability",
            options=['Weekends', 'Weekdays', 'Evenings', 'Mornings', 'Flexible'],
            default=['Flexible']
        )
    
    with col2:
        st.subheader("Additional Information")
        exam_date = st.date_input("When is your target exam date?")
        
        hours_weekly = st.slider(
            "How many hours can you dedicate weekly?",
            min_value=1, 
            max_value=20,
            value=5
        )
        
        specific_goals = st.text_area(
            "Any specific goals or challenges?",
            placeholder="E.g., I struggle with logical reasoning and need help with time management"
        )
    
    if st.button("Find My Mentors"):
        with st.spinner("Finding your perfect mentors..."):
            user_preferences = {
                'preferred_subject': preferred_subject,
                'secondary_subject': secondary_subject,
                'target_college': target_college,
                'learning_style': learning_style,
                'current_level': current_level,
                'availability': availability_pref,
                'hours_weekly': hours_weekly,
                'specific_goals': specific_goals
            }
            
            try:
                recommended_mentors = get_mentor_recommendations(
                    user_preferences, 
                    mentors_df, 
                    encoder, 
                    encoded_mentors, 
                    feature_names
                )
                
                st.subheader("Your Recommended Mentors")
                
                for i, (index, mentor) in enumerate(recommended_mentors.iterrows()):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        st.image(f"https://api.dicebear.com/7.x/initials/svg?seed={mentor['name']}", width=100)
                        st.write(f"**Match: {mentor['match_percentage']:.1f}%**")
                        
                    with col2:
                        st.subheader(mentor['name'])
                        st.write(f"**Expertise:** {mentor['strong_subjects']} | **Also teaches:** {mentor['secondary_subjects']}")
                        st.write(f"**From:** {mentor['alma_mater']} | **CLAT Rank:** {mentor['clat_rank']}")
                        st.write(f"**Teaching Style:** {mentor['teaching_style']} | **Experience:** {mentor['years_experience']} years")
                        st.write(f"**Rating:** {'⭐' * int(round(mentor['rating']))} ({mentor['rating']})")
                        st.write(f"**Availability:** {mentor['availability']}")
                        st.write(f"**Bio:** {mentor['bio']}")
                        st.button(f"Connect with {mentor['name'].split()[0]}", key=f"connect_{i}")
                    
                    st.divider()
                
                st.success("These mentors were selected based on your preferences. You can reach out to them for personalized guidance!")
                
                with st.expander("How we matched you"):
                    st.write("""
                        Our recommendation system uses several factors to match you:
                        1. **Subject expertise alignment** - Prioritizing mentors who excel in your areas of interest
                        2. **Learning style compatibility** - Matching your preferred learning style with mentors' teaching approach
                        3. **Target college expertise** - Selecting mentors from your target institutions
                        4. **Availability** - Ensuring schedules align
                        
                        The system will improve over time as we collect more data on successful mentor-mentee relationships!
                    """)
            except Exception as e:
                st.error(f"An error occurred while generating recommendations: {str(e)}")
                st.info("Please try again or contact support if the problem persists.")

# CLAT Assistant UI
with tab2:
    st.header("CLAT Exam Assistant")
    st.write("Ask me any questions about CLAT exams, preparation, syllabus, etc.")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_query = st.chat_input("Ask about CLAT...")
    
    # Process user input
    if user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        try:
            # Get response from knowledge base with error handling
            response = get_response(user_query, knowledge_base)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
        except Exception as e:
            error_message = "I'm sorry, I encountered an error processing your request. Please try again or ask a different question."
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            
            with st.chat_message("assistant"):
                st.write(error_message)
                st.error(f"Error details: {str(e)}")
    
    # Sample questions
    with st.expander("Sample questions you can ask"):
        st.write("""
        - What is the syllabus for CLAT 2025?
        - How many questions are there in the English section?
        - What's the exam pattern for CLAT?
        - What are the last year's cut-offs for NLSIU Bangalore?
        - How should I prepare for the Legal Reasoning section?
        - What are the important dates for CLAT 2025?
        - Which are the top colleges accepting CLAT scores?
        - What books should I refer to for CLAT preparation?
        """)

# Additional information
st.sidebar.title("About NLTI CLAT Assistant")
st.sidebar.info("""
This app is designed to help CLAT aspirants in their preparation journey.

**Features:**
- Get personalized mentor recommendations based on your learning style and goals
- Ask questions about CLAT exam and get instant answers
- Access valuable resources and tips for preparation

**Need more help?**
Contact NLTI support at support@nlti.in
""")

st.sidebar.divider()
st.sidebar.subheader("System Information")
st.sidebar.write("Version: 1.0.0")
st.sidebar.write("Last updated: April 2025")

# Feedback section
st.sidebar.divider()
st.sidebar.subheader("Your Feedback")
feedback = st.sidebar.text_area("Help us improve", placeholder="Share your experience with this tool...")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback!")

# Resource links
st.sidebar.divider()
st.sidebar.subheader("Additional Resources")
st.sidebar.markdown("""
- [CLAT Official Website](https://consortiumofnlus.ac.in/)
- [NLTI Preparation Programs](https://example.com)
- [Free CLAT Mock Tests](https://example.com)
- [CLAT Preparation Blog](https://example.com)
""")