__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Check if Streamlit is installed correctly, print version
import streamlit as st
print(st.__version__)

# 1: Seeing your data in Streamlit with st.write
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import openai
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.few_shot import DynamicFewShotGPTClassifier
from skllm.models.gpt.text2text.summarization import GPTSummarizer


#Other Libraries
from bs4 import BeautifulSoup
from unidecode import unidecode
import re
import warnings
import joblib
import pickle

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
    
st.set_page_config(layout='wide')

api_key = st.secrets["api_key"]
SKLLMConfig.set_openai_key(api_key)
client = OpenAI(api_key=api_key)

# Constants
CHROMA_DATA_PATH = 'github_embedding'
COLLECTION_NAME = 'github_embedding'

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

## chromadb openai wrapper for embedding layer
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-ada-002")

# Create or get the collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"}
)
################FOR STREAMLIT APP#################################

## filtering function (fix this) -- create filter parameters as part of function

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions


#### Filtering function from Part 1A ####
def choosing_filter(filter_list):     # must pass a dictionary enclosed in a list
  if len(filter_list) > 1:        # two or more filtering options chosen
    where_statement = {"$and": filter_list}
    return where_statement
  elif len(filter_list) == 1:     # one filtering option chosen only
    where_statement = filter_list[0]
    return where_statement
  else: # no filtering (0 in list)
    return None


#### Returning top suggestions function from 1B ####
def return_top_results(query_result):
  ids = query_result.get('ids')[0]
  metadatas = query_result.get('metadatas')[0]
  top_results = list(zip(ids, metadatas))

  suggestions = {}

  for index, results in enumerate(top_results):
    ids = results[0]
    metadatas = results[1]
    suggestion_key = f"suggestion_{index+1}"
    suggestions[suggestion_key] = (ids, metadatas)

  return suggestions


### Function for info fed to ChatGPT ###
def gpt_info(suggestions):
    output = ""
    for index, suggestion_num in enumerate(suggestions):
        output += f'''
{index+1}. {suggestions.get(suggestion_num)[1].get('course_title')}
Description: {suggestions.get(suggestion_num)[1].get('course_description')}
'''
    return output


### Function for printing course suggestions ###
def get_course_suggestions_output(suggestions):
  output = ""
  for index, suggestion_num in enumerate(suggestions):
    output += f'''
-----------------------
**{index+1}. {suggestions.get(suggestion_num)[1].get('course_title')}**\n

Organization: {suggestions.get(suggestion_num)[1].get('course_organization')}\n
Difficulty: {suggestions.get(suggestion_num)[1].get('course_difficulty')}\n
Rating: {suggestions.get(suggestion_num)[1].get('course_rating')}\n
Description: {suggestions.get(suggestion_num)[1].get('course_description')}\n
URL: {suggestions.get(suggestion_num)[1].get('course_url')}\n
'''
  return index+1, output





### Final Chatbot Pipeline

def chatbot_pipeline_input_embedding(query_text, n_results=3, filter_list=[]):
    # Validate query text
    if not query_text.strip():
        st.markdown("Please enter a valid query text.")
        return

# try:
    # Generate embedding for the query text
    client = OpenAI(api_key = api_key)
    query_embedding_response = client.embeddings.create(input=[query_text], model="text-embedding-ada-002")
    query_embedding = query_embedding_response.data[0].embedding


    # where statement from filter list
    where_statement = choosing_filter(filter_list)

    # Query the collection with the generated embedding
    # where parameter filters according to different metadata keys

    if where_statement is not None:
      query_result = collection.query(query_embeddings=[query_embedding], n_results=n_results, where = where_statement, include=['documents','embeddings','metadatas', 'distances'])
    else:
      query_result = collection.query(query_embeddings=[query_embedding], n_results=n_results, include=['documents','embeddings','metadatas', 'distances'])

    # print("\nQuery Result Data:")
    if query_result and 'ids' in query_result and len(query_result['ids'][0]) > 0:

      suggestions = return_top_results(query_result)    # responsible for doing the top course found until the recommendation
      assistant_content_message = gpt_info(suggestions)

      messages = [
          #########################################  Feel free to think of other system-content messages to generate different response from chatgpt #########################################
          {"role": "system", "content": f"You are a bot that provides a brief summary explaining why the courses I provided are relevant to the user's goals, without listing each course individually. Focus on the overall benefits and relevance of the course selection."},
          #########################################  Feel free to think of other system-content messages to generate different response from chatgpt #########################################

          {"role": "user", "content": query_text},
          {"role": "assistant", "content": f'''Here is background information of the courses that you will recommend to the user: {assistant_content_message}'''}
      ]

      client = OpenAI(api_key = api_key)
      response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=messages,
          max_tokens=200  # Allowing enough tokens for a detailed yet concise recommendation
      )

      results_found, course_suggestions = get_course_suggestions_output(suggestions)

      chatbot_response = f'''Here are the top {results_found} course/s that I recommend you to take:
{course_suggestions}
-------------------------------------
{response.choices[0].message.content}
      '''


      return st.subheader(chatbot_response)       # message after recommendation line

    else:
        st.markdown("No results found for the given query.")

    
###---------------pages code-------------------------

# Add page options in a sidebar
my_page = st.sidebar.radio('Page Navigation',
                           ['About our App', 'Exploratory Data Analysis', 
                            'How does the Model work?', 
                            'Course Recommender App'
                           ])


if my_page == 'About our App':
    
    st.subheader('Eskwelabs Data Science Fellowship Cohort 13: Sprint 4')
    st.markdown('A Project by: Alfonso Luis Kan, Carlos Lester Covarrubias II, Jacob Noel Fuentebella, Jules Juanites, and Smart Krys Lazam')
    st.image('streamlit_photos/coursera_header.png', width = 900)
    st.markdown("""
    <style>
        .style-text {
            font-size: 18px;
            line-height: 1.6;
            text-align: justify;
        }
        .highlight {
            font-weight: bold;
            color: #0054D2;
        }
        .margin-bottom {
        margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="style-text">This Streamlit app makes use of Coursera data to develop a machine learning model that can assist you in deciding the perfect online learning course fit to your needs.</p>', unsafe_allow_html=True)

    st.header("Primary Learner Goals")
    st.subheader("Welcome to the Future of Learning!")
    st.markdown('<p class="style-text">Are you looking for:</p>', unsafe_allow_html=True)
    st.markdown('<ul class="style-text"><li>Career benefits</li><li>Personal growth</li><li>Workforce development</li></ul>', unsafe_allow_html=True)
    st.markdown('<p class="style-text">You\'re not alone! Did you know that a whopping <span class="highlight">85% of learners</span> are career-focused? They\'re eager to acquire new skills to get hired, enhance the ones they have, or even pivot to exciting new roles. But, with so many options out there, how do you choose the right path?</p>', unsafe_allow_html=True)

    st.divider()

    st.header("Why :blue[Coursera?]")

    st.markdown('<p class="style-text">Why did we choose Coursera? It\'s simple! Coursera boasts a vast, global network of top-notch courses accessible from anywhere in the world. Whether you\'re looking to learn from <span class="highlight">Ivy League professors</span> or <span class="highlight">industry leaders</span>, Coursera has you covered.</p>', unsafe_allow_html=True)
    st.markdown('<p class="style-text">With Coursera+, you\'ll have the world of knowledge at your fingertips, perfectly curated just for you.</p>', unsafe_allow_html=True)

    st.divider()

    st.header("Our Objectives")
    st.markdown('<div class="margin-bottom"><span class="highlight">Target Audience:</span> Our focus is on Coursera subscribers who are actively seeking courses that resonate with their specific interests and preferences.</div>', unsafe_allow_html=True)
    st.markdown('<div class="margin-bottom"><span class="highlight">User Input:</span> We\'ll be gathering learning interests directly from users. They\'ll simply tell us what they want to learn about, using statements like \'I want to learn about xxx\'.</div>', unsafe_allow_html=True)
    st.markdown('<div class="margin-bottom"><span class="highlight">Personalized Recommendations:</span> To meet these needs, we\'re implementing a cutting-edge system using RAG and Streamlit. This system will intelligently curate and deliver customized course suggestions based on each user\'s unique input.</div>', unsafe_allow_html=True)
    st.markdown('<p class="style-text">Together, these objectives form the backbone of our mission to enhance the learning experience on Coursera, ensuring every subscriber finds exactly what they\'re looking for.</p>', unsafe_allow_html=True)

    st.divider()

    st.header("All 'bout that _Data_")
    st.markdown('<p class="style-text">We\'re leveraging the extensive <span class="highlight">Coursera Course Dataset from 2021</span>, sourced from Kaggle. This robust dataset encompasses over 3,500 courses, each detailed with key attributes including:</p>', unsafe_allow_html=True)
    st.markdown('<ul class="style-text"><li class="margin-bottom">Course Name</li><li class="margin-bottom">Affiliated University</li><li class="margin-bottom">Difficulty Level</li><li class="margin-bottom">User Ratings</li><li class="margin-bottom">URLs for Direct Access</li><li class="margin-bottom">Comprehensive Descriptions</li><li class="margin-bottom">Highlighted Skills</li></ul>', unsafe_allow_html=True)

    st.divider()

    st.header("Just some Limitations~")
    st.markdown('<p class="style-text"><span class="highlight">Data Source:</span> Our initial dataset is sourced from the 2021 Coursera Course Dataset on Kaggle. While extensive, our recommendations will be based on the courses available within this dataset.</p>', unsafe_allow_html=True)
    st.markdown('<p class="style-text"><span class="highlight">Language and Regional Limitations:</span> The dataset and the recommendations may be biased towards courses available in certain languages or regions, since we are only focusing on English courses, which potentially limits recommendations for users seeking courses in less common languages or specific geographical areas.</p>', unsafe_allow_html=True)
    st.markdown('<p class="style-text"><span class="highlight">Accuracy and Coverage:</span> The recommendations generated will depend on the accuracy and coverage of the data and our recommendation model. We aim to optimize accuracy but acknowledge the inherent limitations of any recommendation system.</p>', unsafe_allow_html=True)
        
elif my_page == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis')
    st.markdown('Allow us to take you through the findings we obtained from our data')
    st.divider()

    st.header('Pillars of :blue[Coursera’s] Education')
    st.subheader('Introducing Data, Python, and Management')
    st.image('streamlit_photos/wordcloud_title.png', caption = 'The most common words in the course title\'s', width = 800)
    st.markdown("""
    The word cloud highlights a focus on :blue[**"Data"**], indicating data analysis and science, :blue[**"Python"**], representing programming, and :blue[**"Design"**], suggesting system or software design.
    Showing the demand for technical skills in data handling and software creation.

    Alternatively, the word cloud also highlights the Modern Business Curriculum, which are :blue[**"Management"**] - a key business concept, :blue[**"Analysis"**] - reflecting analytical skills, and :blue[**"Innovation"**] - indicating forward-thinking approaches.

    Technical skills and business concepts are evident in Course titles. Moreover, the word :blue[**“Introduction”**] is highlighted, implying that the majority of the courses are at the beginner difficulty.
    """)
    st.divider()

    st.header('Titans of Skills')
    st.subheader('Leadership Reigns Supreme, Tech Skills Rise')
    st.image('streamlit_photos/wordcloud_skills.png', caption = 'The most common words in the course skill description\'s', width = 800)
    st.markdown("""
    For this Course Skills' Word cloud, it shows Prominent Themes such as :blue[**"Management"**], :blue[**"Leadership"**], :blue[**"Data Science"**], :blue[**"Machine Learning"**], and :blue[**"Computer Science"**], indicating that these are the most emphasized skills across the courses. This suggests a high demand for knowledge in these areas.

    Another key point is Business and Project Management: The frequent appearance of terms like :blue[**"Project Management"**], :blue[**"Business Strategy"**], and :blue[**"Leadership"**] underscores the importance of managerial and strategic skills in 2021's job market

    Lastly, Emerging Trends: Such as :blue[**"Computer Programming"**], :blue[**"Machine learning"**] are gaining importance in the modern educational landscape, since the data is 2021, which as of today, validates the point.

    """)
    st.image('streamlit_photos/top5_skills.png', caption = 'The top 5 skills based on the skills listed', width = 800)
    st.markdown("""
    To have a deeper dive in the Course Skills’ Word Cloud, we have here the Top 5 Most Frequent Skills in the Coursera, :blue[**“project”**], :blue[**“Leadership and Management”**], :blue[**“analysis”**], :blue[**“Computer Programming”**], and :blue[**“Machine Learning”**]
    """)
    st.divider()

    st.header('Coursera’s Commitment')
    st.subheader('Majority of Ratings Soar Above 4.5')
    st.image('streamlit_photos/course_rating.png', caption = 'Most of the courses are rated high, above 4.5 out of 5', width = 800)
    st.markdown("""
    This histogram illustrates the distribution of course ratings on Coursera for the year :blue[**2021**]. The majority of courses are highly rated, with a significant concentration around the :blue[**4.5 to 5.0 range**], indicating overall satisfaction among learners. The density plot overlay highlights a sharp peak, reinforcing the trend of high ratings and showcasing Coursera's commitment to quality education
    """)
    st.divider()

    st.header('Difficulty Meets Satisfaction')
    st.image('streamlit_photos/ratingdifficulty_boxplot.png', caption = 'Course Difficulty and Satisfaction', width = 800)
    st.markdown("""
    This Box Plot, Captures the essence of comparing difficulty levels with course ratings, implying that the relationship between challenge and enjoyment in learning is being explored. It compares course ratings across four difficulty levels: :blue[**Beginner, Advanced, Intermediate, and Conversant.**] Generally, all difficulty levels show high median ratings between :blue[**4.5 and 5.0.**] Interestingly, Advanced courses have the widest range of ratings, including some very low outliers, while Conversant courses show the most consistent ratings with the smallest interquartile range.
    """)
   
elif my_page == 'How does the Model work?':
    st.title('How does the Model work?')
    st.divider()

    st.header('Firstly, Preprocessing')
    st.image('streamlit_photos/prepro.png', width = 800)
    st.markdown("""
    - **Character Encoding**: Removed special characters.
    - **Language Detection**: Filtered out non-English text.
    - **Data Preprocessing**: Dropped duplicates and missing values.
    - **Text Preprocessing**: Removed HTML tags and noise words, tokenized and lemmatized the text.
    - **Data Selection for Modeling**: Disregarded rows labeled as 'Not Calibrated' under 'Course Difficulty' and 'Course Rating'.
    """)
    st.divider()

    st.header('The Modeling and Prediction Pipeline')
    st.markdown('''Today, I'll walk you through our comprehensive end-to-end methodology for our RAG workflow. We'll start from the initial dataset preparation and follow through to the final stage where our chatbot generates a tailored course suggestion in response to a user prompt.''')
    st.image('streamlit_photos/15.png')

    st.markdown('''
    During our data prep phase, we first remove entries without ratings or difficulty levels to maintain data quality. We then clean up the text by removing unnecessary symbols and HTML tags.

    For each course, we utilize its title and description for embedding it into the vector database.

    Additionally, we store relevant course details, such as ratings and difficulty levels, in metadata formatted as a dictionary. This ensures better filtering capabilities when responding to user prompts.
    ''')
    st.image('streamlit_photos/16.png')

    st.markdown('''
    We embed the user query to fetch the top n most similar results from our knowledge base. Filters can be applied as needed to meet user preferences.
    ''')
    st.image('streamlit_photos/17.png')

    st.markdown('''
    After obtaining the course suggestions, we pass this information to ChatGPT and instruct it to provide a concise summary of the course selection. This summary also talks about how the suggested courses can be beneficial to the user based on their entered prompt.
    ''')
    st.image('streamlit_photos/18.png')


elif my_page == 'Course Recommender App':
    st.title('Course Recommender App')
    st.markdown('DISCLAIMER: This online version of the app is limited to only 300 embedded data points due to file size restrictions on the GitHub cloud. Results may not be as accurate.')

    df = pd.read_csv("cleaned_coursera_v2.csv")

    ####filters
    st.sidebar.divider()
    st.sidebar.header('Search Filters')

    #n_results
    n_results = st.sidebar.select_slider(
    "How many results do you want to see?",
    options=[1,2,3,4,5])

    filters = []
    
    #course rating
    rating = st.sidebar.slider("Courses should have a rating above: ", 0.0, 5.0, 0.5,0.1)
    rating_filter = {"course_rating": {"$gte": rating}}
    if rating is not None:
        filters.append(rating_filter)
    # st.write(rating_filter)

    #course difficulty
    difficulty_options = df['Difficulty Level'].unique()
    difficulty = st.sidebar.selectbox("Select Course Difficulty",difficulty_options, index=None)
    difficulty_filter={"course_difficulty": {'$eq': difficulty}}
    if difficulty is not None:
        filters.append(difficulty_filter)
        filtered_df = df[df['Difficulty Level']==difficulty]
        unique_options = sorted(filtered_df['University'].unique())
    else:
        unique_options = sorted(df['University'].unique())
    # st.write(difficulty_filter)
    
    #course orgranization
    
    org = st.sidebar.selectbox("Select Course Provider:", unique_options, index=None)
    #add if-else loop to catch None values
    org_filter={"course_organization": {'$eq': org}}
    if org is not None:
        filters.append(org_filter)
    # st.write(org_filter)

    #user input
    user_input=''
    
    # Add text input for the user input
    user_input = st.text_input(
        label='Do you have any learning goals or a topic you want to learn about?',
        value=''
    )

    if st.button("Search"):
        st.write(f"You searched: {user_input}")
        st.divider()

        # chatbot_pipeline_input_embedding(query_text='I want to build a website in Python and AI', n_results=5, filter_list=[])
        chatbot_pipeline_input_embedding(query_text=user_input, n_results=n_results, filter_list=filters)
       
