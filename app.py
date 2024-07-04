import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


from utils import get_conversational_chain, get_pdf_text, get_text_chunks, get_vector_store



def user_input(user_question):
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
        st.write("Reply: ", response["output_text"])

    except (FileNotFoundError, IOError) as e:
        # Handle file-related errors (missing FAISS index or embedding model)
        print(f"Error: {e}")
        st.error(f"An error occurred while processing your request: {e}")
        # Consider providing user-friendly guidance, e.g., "The system is currently unavailable. Please try again later.")

    except Exception as e:  # Catch more general exceptions
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
        st.error(f"An unexpected error occurred: {e}")
        # Consider logging errors for debugging or alerting administrators




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



# if __name__ == "__main__":
#     main()