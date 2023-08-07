import streamlit as st
import sqlite3
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

DATABASE_NAME = "resumes.db"
GENERATED_QUESTIONS_DB_NAME = "generated_questions.db"
OPENAI_API_KEY =""
def create_resume_database():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS resumes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, resume_text TEXT, similarity_percentage REAL, reading_time FLOAT, file_type TEXT)''')
    conn.commit()
    conn.close()

def save_resume_to_database(resume_text, similarity_percentage, reading_time, file_type):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()

    # Check if a record with the same similarity_percentage and file_type already exists
    c.execute("SELECT * FROM resumes WHERE similarity_percentage = ? AND file_type = ?", (similarity_percentage, file_type))
    existing_record = c.fetchone()

    if existing_record:
        # You can choose to update the existing record here if desired
        # For example:
        # existing_id = existing_record[0]
        # c.execute("UPDATE resumes SET resume_text = ? WHERE id = ?", (resume_text, existing_id))
        pass
    else:
        # Insert a new record if no existing record with the same similarity_percentage and file_type
        c.execute("INSERT INTO resumes (resume_text, similarity_percentage, reading_time, file_type) VALUES (?, ?, ?, ?)", (resume_text, similarity_percentage, reading_time, file_type))

    conn.commit()
    conn.close()

def create_generated_questions_database():
    conn = sqlite3.connect(GENERATED_QUESTIONS_DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS generated_questions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT, answer TEXT)''')
    conn.commit()
    conn.close()

def save_generated_question_and_answer(question, answer):
    conn = sqlite3.connect(GENERATED_QUESTIONS_DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO generated_questions (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()
    conn.close()

def create_credentials_database():
    conn = sqlite3.connect("credentials.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS credentials
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_name TEXT, candidate_email TEXT, phone_number REAL)''')
    conn.commit()
    conn.close()

def save_credentials_to_database(candidate_name, candidate_email, phone_number):
    conn = sqlite3.connect("credentials.db")
    c = conn.cursor()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_reading_time(text):
    num_words = len(text.split())
    reading_time = num_words * 0.008  # Assuming average reading speed of 200 words per minute
    return reading_time

def get_missing_keywords(reference_text, uploaded_text):
    if reference_text is None or reference_text.strip() == "":
        return []  # Return an empty list if reference_text is empty

    vectorizer = CountVectorizer().fit([reference_text])
    keywords = vectorizer.get_feature_names_out()

    # Tokenize uploaded_text (resume)
    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=uploaded_text,
        max_tokens=100,
        stop=None
    )
    tokens = response['choices'][0]['text'].split()[:-1]

    # Find missing keywords
    missing_keywords = [kw for kw in keywords if kw not in tokens]
    return missing_keywords

def generate_questions_openai(job_description):
    openai.api_key = OPENAI_API_KEY
    prompt = f"Generate questions from leetcode,interviewbit,codeforces,code-chef,geeksforgeeks realted to the frameworks mentioned in the following job description:\n{job_description}\n\nQ:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        stop=["Q:"]
    )
    questions = response['choices'][0]['text'].split("\n")
    return questions

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def get_similarity_percentage(reference_text, uploaded_text):
    vectorizer = CountVectorizer().fit_transform([reference_text, uploaded_text])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)
    similarity_percentage = similarity[0][1] * 100
    return similarity_percentage

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def get_top_resumes_from_database():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM resumes WHERE similarity_percentage > 40 ORDER BY similarity_percentage DESC LIMIT 20")
    top_resumes = c.fetchall()
    conn.close()
    return top_resumes

def send_acceptance_email(to_email):
    smtp_server = ""  # Replace with your SMTP server address
    smtp_port = 587  # Replace with your SMTP server port
    smtp_username =  "" # Replace with your SMTP username
    smtp_password = "" # Replace with your SMTP password

    sender_email = ""  # Replace with your sender email
    subject = 'Congratulations! You are accepted!'
    body = 'Dear candidate, \n\nCongratulations! We are pleased to inform you that you have been accepted for the position. \n\nBest regards,\nThe Hiring Team'

    # Create a MIMEText object to represent the email
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = to_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        # Connect to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Start TLS encryption
        server.login(smtp_username, smtp_password)

        # Send the email
        server.sendmail(sender_email, to_email, message.as_string())
        server.quit()
        return True
    except Exception as e:
        print("Failed to send email:", str(e))
        return False
 
def main():
    load_dotenv()
    st.set_page_config(page_title="Techrooter", page_icon="📈")
    st.write(css, unsafe_allow_html=True)

    # Create the databases
    create_resume_database()
    create_generated_questions_database()
    create_credentials_database()

    st.header("Techrooter📈")

    # Define raw_text outside the button block
    raw_text = None
    pre_existing_text = None 
    with st.sidebar:
        st.subheader("Your Resume")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        st.subheader("Job Description")
        pre_existing_doc = st.file_uploader(
            "Upload a Job Description for comparison", type=['txt'])

        similarity_percentage = None

        if st.button("Process") and pdf_docs and pre_existing_doc:
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # Compare uploaded documents to Job Description
                pre_existing_text = pre_existing_doc.read().decode('utf-8')
                similarity_percentage = get_similarity_percentage(
                    pre_existing_text, raw_text)
                st.write(f"Similarity Percentage: {similarity_percentage:.2f}%")

                # Get reading time
                reading_time = get_reading_time(raw_text)
                st.write(f"Estimated Reading Time: {reading_time:.2f} minutes")

                # Save resume to database
                save_resume_to_database(
                    raw_text, similarity_percentage, reading_time, "PDF")

    

    # Check if the resume is available before displaying the text_area for editing
    
    st.subheader("Your Submitted Resume")
    edited_resume = st.text_area(
    "Edit Your Resume", raw_text)  # Use raw_text here instead of st.session_state.edited_resume


    # if "edited_resume" not in st.session_state:
    #         st.session_state.edited_resume = ""


    # Check for missing keywords
    missing_keywords = get_missing_keywords(
            pre_existing_text, edited_resume)
    if missing_keywords:
            st.subheader("Missing Keywords")
            st.write(", ".join(missing_keywords))
    else:
            st.subheader("No Missing Keywords")

  
    # Ask the candidate to continue or keep editing the resume
    continue_editing = st.button("Continue Editing")
    submit_resume = st.button("Submit Resume")

    if continue_editing:
            st.session_state.edited_resume = edited_resume
            st.experimental_rerun()

    if submit_resume:
            pre_existing_text = pre_existing_doc.read().decode('utf-8')
            new_similarity_percentage = get_similarity_percentage(pre_existing_text, edited_resume)
            new_reading_time=get_reading_time(edited_resume)
            save_resume_to_database(edited_resume, new_similarity_percentage, new_reading_time, "PDF")
            # ... (rest of the code)
    # else:
    #         st.warning("Please upload a job description before submitting.")
            

            # Display the updated similarity score
            st.write(f"Updated Similarity Percentage: {new_similarity_percentage:.2f}%")

            # Display form to gather additional information
            st.subheader("Candidate Details")
            candidate_name = st.text_input("Your Name:")
            candidate_email = st.text_input("Your Email:")
            candidate_phone = st.text_input("Your Phone Number:")

            # Submit the information to the credentials database
            if st.button("Submit Information"):
                save_credentials_to_database(candidate_name, candidate_email, candidate_phone)

                st.success("Information submitted successfully!")

                # Send acceptance email to the candidate
                if send_acceptance_email(candidate_email):
                    st.write(f"Acceptance email sent to {candidate_email}")
                else:
                    st.write("Failed to send the acceptance email. Please check your SMTP configuration.")

           

if __name__ == '__main__':
    main()
