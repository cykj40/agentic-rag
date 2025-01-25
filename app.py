import streamlit as st
from helper import check_environment, initialize_llama_index, create_chat_engine, visualize_document
import os

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None

def main():
    # Basic page config
    st.set_page_config(page_title="Blueprint Analyzer", layout="wide")
    
    # Main title
    st.title("Blueprint Analysis Tool")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Document Control")
        
        # Check documents folder
        docs_path = "./documents"
        if not os.path.exists(docs_path):
            os.makedirs(docs_path)
        
        # Show existing files
        files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
        if files:
            st.success(f"Found {len(files)} documents")
            for file in files:
                st.write(f"📄 {file}")
        else:
            st.warning("No PDF files found")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
        if uploaded_file:
            file_path = os.path.join(docs_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")
            
        # Process button
        if st.button("Process Documents"):
            with st.spinner("Processing documents... This may take a few minutes for large PDFs"):
                try:
                    check_environment()
                    with st.status("Processing documents...") as status:
                        status.write("Initializing...")
                        index = initialize_llama_index()
                        if index:
                            status.write("Creating chat engine...")
                            st.session_state.chat_engine = create_chat_engine(index)
                            status.update(label="Done!", state="complete")
                            st.success("Documents processed successfully!")
                        else:
                            st.error("Failed to process documents. Check the logs for details.")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    # Chat interface
    with col1:
        st.header("Chat")
        if st.session_state.chat_engine:
            # Display chat messages
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about the blueprint..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get and display response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chat_engine.chat(prompt)
                        st.write(response.response)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response.response}
                        )
        else:
            st.info("Upload and process documents to start chatting")
    
    # Document viewer
    with col2:
        st.header("Document Viewer")
        if files:
            selected_file = st.selectbox("Select document", files)
            if selected_file:
                file_path = os.path.join(docs_path, selected_file)
                pages = visualize_document(file_path)
                if pages:
                    page_num = st.number_input(
                        "Page", min_value=1, max_value=len(pages), value=1
                    ) - 1
                    st.image(pages[page_num], use_column_width=True)
                else:
                    st.error("Failed to load document")
        else:
            st.info("Upload a PDF to view it here")

if __name__ == "__main__":
    main()
