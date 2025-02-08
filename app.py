import streamlit as st
from helper import initialize_index, create_chat_engine, create_visualization, create_financial_visualization
import os
import fitz
from PIL import Image
import pandas as pd
import json

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None
    if 'current_doc_metadata' not in st.session_state:
        st.session_state.current_doc_metadata = None

def display_pdf(file_path):
    """Display a PDF file in Streamlit."""
    try:
        with fitz.open(file_path) as doc:
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img)
            return pages
    except Exception as e:
        print(f"Error displaying PDF {file_path}: {str(e)}")
        return None

def display_financial_content(metadata):
    """Display financial content extracted from the document."""
    if not metadata or 'financial_data' not in metadata:
        return
    
    fin_data = metadata['financial_data']
    
    # Display financial metrics if found
    if fin_data['metrics']:
        st.subheader("📊 Key Financial Metrics")
        metrics_df = pd.DataFrame(fin_data['metrics'], columns=['Metric'])
        st.dataframe(metrics_df)
    
    # Display financial ratios if found
    if fin_data['ratios']:
        st.subheader("📈 Financial Ratios")
        ratios_df = pd.DataFrame(fin_data['ratios'], columns=['Ratio'])
        st.table(ratios_df)
    
    # Display market data if found
    if fin_data['market_data']:
        st.subheader("💹 Market Data")
        market_df = pd.DataFrame(fin_data['market_data'], columns=['Data Point'])
        st.dataframe(market_df)
    
    # Display dates if found
    if fin_data['dates']:
        st.subheader("📅 Important Dates")
        dates_df = pd.DataFrame(fin_data['dates'], columns=['Date'])
        st.table(dates_df)
    
    # Display tables (financial statements) if found
    if metadata.get('has_tables', False):
        st.subheader("📑 Financial Statements")
        for i, table_dict in enumerate(metadata['tables']):
            st.write(f"Statement {i+1}:")
            df = pd.DataFrame.from_dict(table_dict)
            st.dataframe(df)
            
            # Create visualizations for numerical data
            if len(df) > 1 and df.select_dtypes(include=['number']).columns.any():
                st.write("📊 Trend Analysis:")
                numeric_cols = df.select_dtypes(include=['number']).columns
                for col in numeric_cols:
                    viz_data = df.set_index(df.columns[0])[col]
                    viz_base64 = create_financial_visualization({'Time': viz_data.index, 'Value': viz_data.values}, 'line')
                    if viz_base64:
                        st.image(f"data:image/png;base64,{viz_base64}")

def process_and_display_response(response_text, metadata=None):
    """Process and display the AI response with financial analysis and visualizations."""
    # Split response into sections
    sections = response_text.split('\n\n')
    
    for section in sections:
        # Handle tables (financial data)
        if '|' in section and '-|-' in section:
            # Convert markdown table to dataframe
            lines = [line.strip() for line in section.split('\n')]
            headers = [h.strip() for h in lines[0].strip('|').split('|')]
            df = pd.DataFrame([
                [cell.strip() for cell in row.strip('|').split('|')]
                for row in lines[2:]
            ], columns=headers)
            st.table(df)
            
            # Create financial visualization if appropriate
            if len(df) > 1 and df.select_dtypes(include=['number']).columns.any():
                st.write("📈 Financial Visualization:")
                viz_data = df.set_index(df.columns[0])
                # Try different chart types based on data
                for viz_type in ['line', 'bar', 'candlestick']:
                    viz_base64 = create_financial_visualization(viz_data, viz_type)
                    if viz_base64:
                        st.image(f"data:image/png;base64,{viz_base64}")
                        break
        
        # Handle code blocks (calculations)
        elif '```' in section:
            code = section.split('```')[1]
            st.code(code)
        
        # Handle financial formulas
        elif '$' in section:
            st.latex(section.strip('$'))
        
        # Regular text
        else:
            st.write(section)
    
    # Display financial content if available
    if metadata:
        with st.expander("📊 Financial Analysis Details", expanded=False):
            display_financial_content(metadata)

def main():
    # Page config
    st.set_page_config(
        page_title="Financial Document Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session
    init_session_state()

    # Main layout
    st.title("💰 Financial Document Analysis System")
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    # Left Column - Analysis Interface
    with col1:
        st.header("📊 Financial Analysis")
        
        # Document Processing Section
        with st.expander("📁 Document Control", expanded=True):
            # Tabs for different input methods
            tab1, tab2 = st.tabs(["📄 Upload PDF", "📝 Paste Text"])
            
            # PDF Upload Tab
            with tab1:
                uploaded_file = st.file_uploader("Upload Financial Document (PDF)", type=['pdf'])
                if uploaded_file:
                    docs_path = "./documents"
                    if not os.path.exists(docs_path):
                        os.makedirs(docs_path)
                        
                    file_path = os.path.join(docs_path, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Uploaded: {uploaded_file.name}")
            
            # Text Input Tab
            with tab2:
                text_source = st.text_input("Source Name (e.g., 'Q4 2023 Earnings Call')", "")
                text_content = st.text_area(
                    "Paste text content here (e.g., earnings call transcript, financial report, news article)",
                    height=200
                )
            
            # Process button
            if st.button("Analyze Content", type="primary"):
                with st.spinner("Analyzing data..."):
                    try:
                        # Initialize index with both PDF and text content
                        index = initialize_index(
                            text_content=text_content if text_content.strip() else None,
                            text_source=text_source if text_source.strip() else None
                        )
                        if index:
                            st.session_state.chat_engine = create_chat_engine(index)
                            st.success("✅ Analysis complete!")
                            st.rerun()
                        else:
                            st.error("Failed to analyze content")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.error("Make sure your OpenAI API key is set correctly in the .env file")
        
        # Chat Interface
        if st.session_state.chat_engine:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        process_and_display_response(
                            message["content"],
                            message.get("metadata", None)
                        )
                    else:
                        st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about financial metrics, trends, or analysis..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing data..."):
                        response = st.session_state.chat_engine.chat(prompt)
                        metadata = None
                        if hasattr(response, 'source_nodes') and response.source_nodes:
                            metadata = response.source_nodes[0].metadata
                        
                        process_and_display_response(response.response, metadata)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.response,
                            "metadata": metadata
                        })
        else:
            st.info("👆 Upload documents or paste text to start the analysis!")
    
    # Right Column - Document/Content Viewer
    with col2:
        st.header("📄 Content Viewer")
        
        # Show available content
        docs_path = "./documents"
        if os.path.exists(docs_path):
            files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]
            
            if files:
                st.subheader("📑 PDF Documents")
                selected_file = st.selectbox("Select document to view", files)
                if selected_file:
                    file_path = os.path.join(docs_path, selected_file)
                    pages = display_pdf(file_path)
                    if pages:
                        # Page navigation
                        page_num = st.number_input("Page", min_value=1, max_value=len(pages), value=1) - 1
                        
                        # Display current page
                        st.image(pages[page_num], use_column_width=True)
                        
                        # Page info
                        st.caption(f"Page {page_num + 1} of {len(pages)}")
                        
                        # Display financial content for current page
                        if st.session_state.current_doc_metadata:
                            with st.expander("💹 Financial Analysis", expanded=False):
                                display_financial_content(st.session_state.current_doc_metadata)
                    else:
                        st.error("Failed to load document")
            
            if text_content:
                st.subheader("📝 Pasted Text Content")
                with st.expander("View Text Content", expanded=False):
                    st.write(text_content)
                    if st.session_state.current_doc_metadata:
                        st.subheader("💹 Analysis")
                        display_financial_content(st.session_state.current_doc_metadata)
            
            if not files and not text_content:
                st.info("No content available. Upload a PDF or paste text to get started!")
        else:
            st.info("Upload a document or paste text to view it here!")

if __name__ == "__main__":
    main()
