
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3, convert_system_message_to_human=True)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def ask_question(user_question):
    try:
        # Load embeddings model and FAISS index
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # new_db = faiss.read_index("faiss_index")  # Assuming faiss_index exists

        # Find similar documents using the embedding model
        # docs = new_db.search(np.array([embeddings.encode(user_question)]), k=1)[1][0]

        # new_db = FAISS.load_local("faiss_index", embeddings)
        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        # Generate conversational chain using user question and retrieved documents
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        # Print and display response
        print(response)
        return response

    except (FileNotFoundError, IOError) as e:
        # Handle file-related errors (missing FAISS index or embedding model)
        print(f"Error: {e}")
        return response
        # Consider providing user-friendly guidance, e.g., "The system is currently unavailable. Please try again later.")

    except Exception as e:  # Catch more general exceptions
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
        return response
        # Consider logging errors for debugging or alerting administrators



