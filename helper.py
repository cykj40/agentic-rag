import os
import traceback
from dotenv import load_dotenv, find_dotenv
import pytesseract
from PIL import Image
import cv2
import numpy as np

# IMPORTANT: Depending on your LlamaIndex version, the import paths may differ:
# If "Document" is not found in llama_index.core, use: 
# from llama_index import Document
# or from llama_index.schema import Document
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.openai import OpenAIEmbedding

# If you use `langchain-openai`:
from langchain_openai import ChatOpenAI
# Otherwise, if using stock LangChain:
# from langchain.chat_models import ChatOpenAI

import fitz
import io

# Configure Tesseract for Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_env():
    """Load environment variables from .env file."""
    _ = load_dotenv(find_dotenv())


def check_environment():
    """Check if environment is properly configured."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    return True


def process_pdf_to_document(pdf_path: str) -> Document:
    """Process technical PDFs with focus on blueprint content"""
    print(f"Processing technical document: {pdf_path}")
    
    try:
        pdf_document = fitz.open(pdf_path)
        print(f"Document loaded: {len(pdf_document)} pages")
        
        extracted_content = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Get basic text
            text = page.get_text()
            
            # Get technical elements
            drawings = page.get_drawings()
            annotations = page.annots()
            
            # Extract measurements and dimensions
            blocks = page.get_text("blocks")  # Gets text with position info
            
            # Combine page content
            page_content = [f"Page {page_num + 1} Content:"]
            page_content.append(f"Text Content:\n{text}")
            
            if drawings:
                page_content.append(f"Technical Elements: {len(drawings)} drawing elements found")
            
            if annotations:
                annot_text = [a.info.get("content", "") for a in annotations if a.info.get("content")]
                if annot_text:
                    page_content.append(f"Annotations:\n{' '.join(annot_text)}")
            
            if blocks:
                measurements = [b[4] for b in blocks if any(unit in b[4].lower() for unit in ['mm', 'cm', 'm', 'inch', 'ft', '"', "'"])]
                if measurements:
                    page_content.append(f"Measurements found:\n{' '.join(measurements)}")
            
            extracted_content.append("\n".join(page_content))
        
        pdf_document.close()
        
        # Create a rich document with metadata
        return Document(
            text="\n\n".join(extracted_content),
            metadata={
                "source": pdf_path,
                "type": "technical_drawing",
                "pages": len(pdf_document),
                "filename": os.path.basename(pdf_path)
            }
        )
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        traceback.print_exc()
        return None


def build_documents_list(docs_folder: str) -> list:
    """Process all PDFs in the folder"""
    all_docs = []
    print(f"Scanning folder: {docs_folder}")
    
    for fname in os.listdir(docs_folder):
        if fname.lower().endswith('.pdf'):
            fpath = os.path.join(docs_folder, fname)
            print(f"Found PDF: {fname}")
            doc = process_pdf_to_document(fpath)
            if doc:
                all_docs.append(doc)
                print(f"Successfully processed: {fname}")
    
    return all_docs


def initialize_llama_index(documents_path: str = "./documents"):
    """Initialize RAG system with technical document understanding"""
    try:
        # Verify environment
        load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment")
        
        # Process all documents
        print(f"Processing technical documents from: {documents_path}")
        documents = []
        
        for filename in os.listdir(documents_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(documents_path, filename)
                doc = process_pdf_to_document(file_path)
                if doc:
                    documents.append(doc)
                    print(f"Successfully processed: {filename}")
        
        if not documents:
            raise ValueError("No documents were successfully processed")
        
        # Create optimized index
        print("Creating technical document index...")
        llm = ChatOpenAI(
            model="gpt-4",
            api_key=api_key,
            temperature=0.2  # Lower temperature for more precise technical responses
        )
        
        embed_model = OpenAIEmbedding(api_key=api_key)
        
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model
        )
        
        print("Technical document index created successfully")
        return index
        
    except Exception as e:
        print(f"Index creation error: {str(e)}")
        traceback.print_exc()
        return None


def create_chat_engine(index):
    """Create a specialized chat engine for technical documents"""
    return index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True,
        system_prompt=(
            "You are an expert in analyzing technical drawings, blueprints, and engineering documentation. "
            "Focus on providing precise, technical information including measurements, dimensions, and specifications. "
            "When discussing blueprints, reference specific pages and sections. "
            "If you're unsure about any technical detail, say so rather than making assumptions. "
            "Use technical terminology appropriate for architectural and engineering contexts."
        )
    )


def visualize_document(file_path):
    """Convert each page of the PDF into an image (using PyMuPDF/fitz) or show an image file."""
    try:
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap()
                img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img_data)
            return pages
        else:
            # If not PDF, assume it's an image
            return [Image.open(file_path)]
    except Exception as e:
        print(f"Error visualizing {file_path}: {str(e)}")
        traceback.print_exc()
        return None



