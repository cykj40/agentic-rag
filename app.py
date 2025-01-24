import streamlit as st
from helper import check_environment, initialize_llama_index, create_chat_engine, visualize_document
import os
import json
from datetime import datetime
import base64
from PIL import Image
import io

def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if 'current_doc' not in st.session_state:
        st.session_state.current_doc = None

def load_documents():
    """Load and process documents"""
    try:
        check_environment()
        index = initialize_llama_index()
        if index is None:
            st.error("Failed to initialize document index. Check the documents and try again.")
            return False
            
        st.session_state.chat_engine = create_chat_engine(index)
        return True
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return False

def save_uploaded_file(uploaded_file):
    """Save uploaded file to documents folder"""
    try:
        docs_path = "./documents"
        if not os.path.exists(docs_path):
            os.makedirs(docs_path)
        
        file_path = os.path.join(docs_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return False

def export_chat_history():
    """Export chat history as JSON"""
    if not st.session_state.messages:
        return None
    
    export_data = {
        "thread_id": st.session_state.thread_id,
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages
    }
    
    json_str = json.dumps(export_data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    return f'data:application/json;base64,{b64}'

def display_document_viewer():
    """Display document viewer with visualization controls"""
    docs_path = "./documents"
    if not os.path.exists(docs_path):
        return
    
    files = [f for f in os.listdir(docs_path) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
    if not files:
        return
    
    selected_doc = st.selectbox("Select document to view", files)
    if selected_doc:
        st.session_state.current_doc = os.path.join(docs_path, selected_doc)
        
        # Display document visualization
        pages = visualize_document(st.session_state.current_doc)
        if pages:
            # Add page selector
            total_pages = len(pages)
            if total_pages > 1:
                page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1
            else:
                page_num = 0
            
            current_page = pages[page_num]
            st.image(current_page, use_column_width=True)
            
            # Add visualization controls
            col1, col2 = st.columns(2)
            with col1:
                zoom = st.slider("Zoom", 50, 200, 100, 10)
            with col2:
                rotate = st.selectbox("Rotate", [0, 90, 180, 270])
            
            if zoom != 100 or rotate != 0:
                # Apply transformations
                if isinstance(current_page, Image.Image):
                    if zoom != 100:
                        new_size = tuple(int(dim * zoom/100) for dim in current_page.size)
                        current_page = current_page.resize(new_size, Image.Resampling.LANCZOS)
                    if rotate != 0:
                        current_page = current_page.rotate(rotate, expand=True)
                    st.image(current_page, use_column_width=True)

def main():
    st.set_page_config(page_title="Technical Document Analysis", layout="wide")
    
    st.title("Technical Document Analysis Chat")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("Documents")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload technical documents or drawings"
        )
        if uploaded_file:
            if save_uploaded_file(uploaded_file):
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                if st.button("Process New Document"):
                    if load_documents():
                        st.success("Documents reloaded successfully!")
        
        if st.button("Reload All Documents"):
            if load_documents():
                st.success("Documents loaded successfully!")
        
        # Document info
        st.subheader("Supported Files")
        st.write("- PDF files (.pdf)")
        st.write("- Images (.png, .jpg, .jpeg)")
        
        # Show files in documents folder
        docs_path = "./documents"
        if os.path.exists(docs_path):
            files = [f for f in os.listdir(docs_path) if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
            if files:
                st.subheader("Loaded Files")
                for file in files:
                    st.write(f"📄 {file}")
            else:
                st.warning("No supported files found in documents folder")
        
        # Export chat history
        if st.session_state.messages:
            st.download_button(
                "Export Chat History",
                export_chat_history(),
                f"chat_history_{st.session_state.thread_id}.json",
                "application/json"
            )
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["Chat", "Document Viewer"])
    
    with tab1:
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Load documents if chat engine not initialized
            if st.session_state.chat_engine is None:
                if load_documents():
                    st.success("Documents loaded successfully!")
            
            # Thread selector
            if st.session_state.messages:
                if st.button("Start New Thread"):
                    st.session_state.messages = []
                    st.session_state.thread_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.rerun()
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about your technical documents..."):
                if st.session_state.chat_engine is None:
                    st.error("Please add documents to the documents folder first!")
                    return
                
                # Add user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "thread_id": st.session_state.thread_id
                })
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chat_engine.chat(prompt)
                        st.markdown(response.response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.response,
                            "timestamp": datetime.now().isoformat(),
                            "thread_id": st.session_state.thread_id
                        })
    
    with tab2:
        # Document viewer
        display_document_viewer()

if __name__ == "__main__":
    main() 