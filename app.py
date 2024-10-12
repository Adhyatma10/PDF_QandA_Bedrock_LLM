import json
import os
import sys
import boto3
import streamlit as st

# Using AWS Bedrock's Titan Embeddings Model for generating text embeddings
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Data ingestion and processing
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# For Vector Embedding and Vector Store Management (FAISS is used for efficient similarity search)
from langchain_community.vectorstores import FAISS

# LangChain components for creating LLM-based retrieval QA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize AWS Bedrock client using boto3 to interact with Amazon Bedrock services
bedrock = boto3.client(service_name="bedrock-runtime")

# Embedding model using Amazon's Titan for text embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


# Data ingestion: Load PDFs from a directory and process them into text chunks
def data_ingestion():
    # Load all PDF files from the 'data' directory using PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()

    # Use a character-based text splitter to split documents into manageable chunks
    # for embedding generation. In testing, this worked better for large PDF datasets.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    
    # Split the loaded documents into chunks
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Embedding: Create FAISS vector store from document embeddings
def get_vector_store(docs):
    # Use FAISS (an efficient similarity search library) to store the document embeddings
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings  # Embed documents using the Titan embeddings model
    )
    # Save the vector store locally as 'faiss_index' for reuse
    vectorstore_faiss.save_local("faiss_index")

# Function to create Claude v2 LLM client from AWS Bedrock
def get_claude_llm():
    # Use Bedrock to create the Claude v2 LLM from Anthropic
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock)
    return llm

# Function to create Llama2 LLM client from AWS Bedrock
def get_llama2_llm():
    # Use Bedrock to create Meta's Llama2 70B Chat Model
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock)
    return llm

# Custom prompt template for generating responses from the LLM
# The template provides a context and question, and the LLM must generate a detailed 250-word answer
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least 200 words with detailed explanations. 
If you don't know the answer, just say that you don't know, don't make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

# Initialize the PromptTemplate class with the above prompt
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Function to get responses from the LLM based on the query and the FAISS vector store
def get_response_llm(llm, vectorstore_faiss, query):
    # Create a RetrievalQA chain, which retrieves relevant documents from the FAISS vector store
    # and passes them as context to the LLM using the "stuff" chain type (concatenation of context)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}  # Retrieve top 3 similar chunks
        ),
        return_source_documents=True,  # Return source documents used to generate the answer
        chain_type_kwargs={"prompt": PROMPT}  # Use the custom prompt template for answers
    )
    
    # Get the answer based on the query and return the result
    answer = qa({"query": query})
    return answer['result']

# Main function to run the Streamlit app
def main():
    # Set up the Streamlit page configuration with a title
    st.set_page_config("Chat PDF")

    # Page header for the app
    st.header("Chat with PDF using AWS Bedrock")

    # Input field for the user to ask questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Sidebar for vector store management
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        # Button to update or create the vector store from the documents
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                # Ingest and process PDF data
                docs = data_ingestion()
                # Generate and save document embeddings
                get_vector_store(docs)
                st.success("Done")  # Display success message once the process is done

    # Button to get output from Claude v2
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            # Load FAISS vector store
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            # Get Claude v2 LLM instance
            llm = get_claude_llm()
            # Get the response from Claude based on the user's question
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")  # Display success message

    # Button to get output from Llama2
    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            # Load FAISS vector store
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            # Get Llama2 LLM instance
            llm = get_llama2_llm()
            # Get the response from Llama2 based on the user's question
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")  # Display success message

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
