import streamlit as st
from helper import initialize_llama_index, create_chat_engine, visualize_document
import os

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None

def main():
    # Page config
    st.set_page_config(page_title="Blueprint Analyzer", layout="wide")
    
    # Initialize session
    init_session_state()

    # Main layout
    st.title("Blueprint Analysis Tool")
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    # Left Column - Chat Interface
    with col1:
        st.header("💬 Chat with your Blueprints")
        
        # Document Processing Section
        with st.expander("📁 Document Control", expanded=True):
            # File uploader
            uploaded_file = st.file_uploader("Upload Blueprint (PDF)", type=['pdf'])
            if uploaded_file:
                docs_path = "./documents"
                if not os.path.exists(docs_path):
                    os.makedirs(docs_path)
                    
                file_path = os.path.join(docs_path, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded: {uploaded_file.name}")
            
            # Process button
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        index = initialize_llama_index()
                        if index:
                            st.session_state.chat_engine = create_chat_engine(index)
                            st.success("✅ Documents processed successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to process documents")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.error("Make sure your OpenAI API key is set correctly in the .env file")
        
        # Chat Interface
        st.markdown("### Chat")
        if st.session_state.chat_engine:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about your blueprints..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        response = st.session_state.chat_engine.chat(prompt)
                        st.markdown(response.response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.response
                        })
        else:
            st.info("👆 Upload and process your blueprints to start chatting!")
    
    # Right Column - Document Viewer
    with col2:
        st.header("🔍 Document Viewer")
        
        # Show available documents
        docs_path = "./documents"
        if os.path.exists(docs_path):
            files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
            if files:
                selected_file = st.selectbox("Select document to view", files)
                if selected_file:
                    file_path = os.path.join(docs_path, selected_file)
                    pages = visualize_document(file_path)
                    if pages:
                        # Page navigation
                        page_num = st.number_input("Page", min_value=1, max_value=len(pages), value=1) - 1
                        
                        # Display current page
                        st.image(pages[page_num], use_column_width=True)
                        
                        # Page info
                        st.caption(f"Page {page_num + 1} of {len(pages)}")
                    else:
                        st.error("Failed to load document")
            else:
                st.info("No documents available. Upload a PDF to get started!")
        else:
            st.info("Upload a document to view it here!")

if __name__ == "__main__":
    main()
