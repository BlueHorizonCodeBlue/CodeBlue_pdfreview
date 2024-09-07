import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time

# LM Studio Server settings
LM_STUDIO_URL = "http://localhost:1234"  # Adjust port if needed

@st.cache_data
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def query_lm_studio(prompt, max_tokens=512, temperature=0.7):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    response = requests.post(f"{LM_STUDIO_URL}/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        st.error(f"Error querying LM Studio: {response.text}")
        return None

def create_embeddings(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts, embeddings)

def extract_rules_from_style_guides(style_guides):
    rules = []
    for guide in style_guides:
        prompt = f"Extract the key rules and recommendations from the following style guide:\n\n{guide}"
        response = query_lm_studio(prompt, max_tokens=1024)
        if response:
            rules.append(response)
    return rules

def query_llm(document_text, query, rules, agents):
    # Use a more sophisticated text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    document_chunks = text_splitter.split_text(document_text)
    
    # Create embeddings for document chunks
    vector_db = create_embeddings(document_chunks)
    
    responses = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, agent in enumerate(agents):
        status_text.text(f"Processing agent {i+1}/{len(agents)}: {agent['name']}")
        
        try:
            start_time = time.time()
            
            # Retrieve relevant chunks from the document
            relevant_chunks = vector_db.similarity_search(query, k=5)
            context = "\n".join([chunk.page_content for chunk in relevant_chunks])
            
            # Combine rules into a single string
            rules_text = "\n\n".join(rules)
            
            prompt = f"{agent['system_prompt']}\n\nRules and Recommendations:\n{rules_text}\n\nContext:\n{context}\n\nQuery: {query}"
            response = query_lm_studio(prompt, max_tokens=1024)
            
            end_time = time.time()
            
            if response:
                responses.append(f"{agent['name']}: {response}\n(Generated in {end_time - start_time:.2f} seconds)")
            else:
                responses.append(f"{agent['name']}: Failed to generate a response.")
        except Exception as e:
            responses.append(f"{agent['name']}: An error occurred: {str(e)}")
        
        progress_bar.progress((i + 1) / len(agents))
    
    status_text.text("Processing complete!")
    return "\n\n".join(responses)

# Streamlit UI
st.title("Document Style Checker")

# Load Style Guides
st.subheader("Load Style Guides")
style_guide_files = st.file_uploader("Upload up to 5 style guide documents", type="pdf", accept_multiple_files=True)
if st.button("Load Style Guides"):
    with st.spinner("Loading style guides..."):
        style_guides = [process_pdf(file) for file in style_guide_files[:5]]
        rules = extract_rules_from_style_guides(style_guides)
        st.session_state['rules'] = rules
        st.success(f"Loaded and extracted rules from {len(style_guides)} style guide(s)")

# Document upload
st.subheader("Upload Document for Review")
document_file = st.file_uploader("Upload document to review", type="pdf")

# Query input
query = st.text_input("Enter your query about the document:")

# Define agents (customize as needed)
agents = [
    {"name": "Style Checker", "system_prompt": "You are a style guide expert. Analyze the document for style consistency."},
    {"name": "Word Choice Advisor", "system_prompt": "You are a word choice expert. Suggest improvements for word choices."},
    {"name": "Structure Analyst", "system_prompt": "You are a document structure expert. Analyze the document's organization."}
]

if document_file and query and 'rules' in st.session_state:
    if st.button("Analyze Document"):
        with st.spinner("Analyzing document..."):
            document_text = process_pdf(document_file)
            result = query_llm(document_text, query, st.session_state['rules'], agents)
            st.session_state['analysis_result'] = result
            st.subheader("Analysis Results:")
            st.success("Analysis complete! You can now view the results below or download them as a text file.")

# Display results in an expandable format
if 'analysis_result' in st.session_state:
    for agent_result in st.session_state['analysis_result'].split("\n\n"):
        parts = agent_result.split(":", 1)
        agent_name = parts[0]
        agent_content = parts[1] if len(parts) > 1 else "No content provided."
        
        with st.expander(agent_name):
            st.write(agent_content)

    # Provide download option
    st.download_button(
        label="Download Full Analysis",
        data=st.session_state['analysis_result'],
        file_name="document_analysis.txt",
        mime="text/plain"
    )
