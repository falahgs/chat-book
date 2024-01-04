import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma

# Create a folder for storing PDFs
pdfs_folder = "pdfs"
os.makedirs(pdfs_folder, exist_ok=True)
# Streamlit configuration
st.set_page_config(
    page_title="Question Answering with LangChain and Google Generative AI",
    page_icon="🔍",
    layout="centered")
# Add CSS styles
st.markdown(
    """
    <style>
        body {
            direction: ltr;
            text-align: left;
        }
        .center {
            text-align: center;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            right: 10px;
            color: #808080;
            text-align: center;
        }
        .border {
            border: 2px solid #555;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hide unnecessary elements
hide_style = '''
<style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    stDeployButton {display:none;}
    .css-hi6a2p {padding-top: 0rem;}
    head {visibility:hidden;}
</style>
'''
st.markdown(hide_style, unsafe_allow_html=True)

# Main title
st.markdown("<h1 class='title center'>Smart PDF Analysis</h1>", unsafe_allow_html=True)
logo_url="logo.jpg"
welcome_message = """**Welcome to Smart PDF Analysis!**

Unlock the power of artificial intelligence with our cutting-edge solution, combining the prowess of LangChain and Google Generative AI. Seamlessly analyze and extract valuable insights from your PDF documents like never before.

**Key Features:**

- **Intelligent Document Understanding:** Leverage LangChain's advanced question-answering capabilities to derive meaningful information from your PDFs.

- **Google Generative AI Magic:** Harness the creative prowess of Google Generative AI to extract nuanced details and generate insightful responses.

- **Effortless Workflow:** Simply upload your PDF files, and our smart system will handle the rest. Extracting relevant data has never been so intuitive.

**Benefits:**

- **Time-Efficient:** Swiftly analyze PDFs and receive detailed answers to your questions, saving you valuable time.

- **Accurate Insights:** Rely on the combined power of LangChain and Google Generative AI for accurate and reliable document analysis.

- **User-Friendly:** Our platform is designed with simplicity in mind, ensuring a seamless experience for all users.

Experience the future of document analysis with "Smart PDF Analysis: Unleashing LangChain and Google Generative AI." Try it now!

---"""

# Sidebar
with st.sidebar:
    #logo_image = st.sidebar.image("logo.jpg", use_container_width=True)
    st.sidebar.image(logo_url,width=100)
    with st.expander("Welcome"):
     st.write(welcome_message)
    st.subheader("Gemini Pro Settings")
    #st.divider()
    #st.info(
      # "What is meaning CognoCraft? The Cogno is derived from cognition or cognitive, relating to mental processes such as learning, reasoning, and understanding. Craft implies skill and artistry in creating or delivering something. So, CognoCraft could be interpreted as the skillful application of cognitive processes, suggesting intelligence, creativity, and craftsmanship in the field of artificial intelligence and conversation."
   # )


# Streamlit sidebar for API key input
API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
# Load the API key into the environment variable
os.environ['GOOGLE_API_KEY'] = API_KEY
if not API_KEY:
        st.info("Enter the Google API Key to continue")
        st.stop()
# Streamlit main content
#st.title("Question Answering with LangChain and Google Generative AI")
# Streamlit user input for the question
# Streamlit file upload for PDF files
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
question = st.text_input("Enter your question", "")
question=question +" from this book"
# Process uploaded PDFs
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
top_k = st.sidebar.number_input("Top K", value=32)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0)
max_output_tokens = st.sidebar.number_input("Max Tokens", value=1024)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Processing file: {uploaded_file.name}")

        # Save PDF to the "pdfs" folder
        pdf_path = os.path.join(pdfs_folder, uploaded_file.name)
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(uploaded_file.getvalue())

        # Load PDF data
        loader = PyPDFDirectoryLoader(pdfs_folder)
        data = loader.load_and_split()
        context = "\n".join(str(p.page_content) for p in data)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        texts = text_splitter.split_text(context)

        # Check if 'texts' is not empty before creating the Chroma vector index
        if texts:
            # Google Generative AI Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Chroma vector index
            try:
                vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

                # Get relevant documents for the question
                docs = vector_index.get_relevant_documents(question)

                # LangChain prompt template
                prompt_template = """
                give me detailed about my question as detailed as possible from the provided context, from this book, \n\n
                Context:\n {context}?\n
                Detailed: \n{detailed}\n

                Detailed:
                """

                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "detailed"])

                # Google Generative AI model
                model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=temperature,top_k=top_k,top_p=top_p,max_output_tokens=max_output_tokens)

                # Load QA chain
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

                # Execute the QA chain and display the result
                button_key = f"generate_answer_button_{uploaded_file.name}"
                if st.button("Generate Answer", key=button_key):
                    try:
                        response = chain({"input_documents": docs, "detailed": question}, return_only_outputs=True)
                        st.subheader("Generated Answer:")
                        #st.write(response['output_text'])
                        # Split the response into lines and display each line with st.code()
                        #lines = response['output_text'].split('\n')
                        #for line in lines:
                            #st.code(line)
                        st.code(response['output_text'])
                    except Exception as e:
               # st.error(f"Error: {e}", icon="🚨")
                        #st.warning("there is no information about your quesion in this book")
                        st.code("there is no information about your quesion in this book")
            except Exception as e:
               # st.error(f"Error: {e}", icon="🚨")
                        #st.warning("there is no information about your quesion in this book")
                        st.warning("there is error...please retry ")

        else:
            st.warning("No valid text found for processing.")
# Footer
st.sidebar.markdown("<div class='footer'>© Falah.G.Salieh, 2024. All Rights Reserved.</div>", unsafe_allow_html=True)
