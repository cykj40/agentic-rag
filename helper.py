import os
import traceback
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import fitz
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

def process_pdf_to_document(pdf_path: str) -> Document:
    """Process technical PDFs with focus on blueprint content"""
    print(f"Processing technical document: {pdf_path}")
    
    try:
        with fitz.open(pdf_path) as pdf_document:
            page_count = len(pdf_document)
            print(f"Document loaded: {page_count} pages")
            
            extracted_content = []
            for page_num in range(page_count):
                page = pdf_document[page_num]
                
                # Get basic text
                text = page.get_text()
                
                # Get technical elements
                drawings = page.get_drawings()
                annotations = page.annots()
                
                # Extract measurements and dimensions
                blocks = page.get_text("blocks")
                
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
            
            # Create document
            return Document(
                text="\n\n".join(extracted_content),
                metadata={
                    "source": pdf_path,
                    "type": "technical_drawing",
                    "pages": page_count,
                    "filename": os.path.basename(pdf_path)
                }
            )
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        traceback.print_exc()
        return None

def initialize_llama_index(documents_path: str = "./documents"):
    """Initialize RAG system with technical document understanding"""
    try:
        # Load environment
        load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment")

        # Configure LLM and embeddings
        Settings.llm = OpenAI(
            model="gpt-4",
            api_key=api_key,
            temperature=0.7
        )
        Settings.embed_model = OpenAIEmbedding(
            api_key=api_key,
            model="text-embedding-3-small"
        )
        
        # Process documents
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
        
        # Create index
        print("Creating technical document index...")
        index = VectorStoreIndex.from_documents(documents)
        print("Technical document index created successfully")
        return index
        
    except Exception as e:
        print(f"Index creation error: {str(e)}")
        traceback.print_exc()
        return None

def create_chat_engine(index):
    """Create a specialized chat engine for technical documents"""
    return index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        context_template=(
            "You are an expert in analyzing technical drawings, blueprints, and engineering documentation. "
            "Focus on providing precise, technical information including measurements, dimensions, and specifications. "
            "When discussing blueprints, reference specific pages and sections. "
            "If you're unsure about any technical detail, say so rather than making assumptions. "
            "Use technical terminology appropriate for architectural and engineering contexts.\n\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given this context, please respond to the question: {query_str}\n"
        )
    )

def visualize_document(file_path):
    """Convert each page of the PDF into an image"""
    try:
        if file_path.lower().endswith('.pdf'):
            with fitz.open(file_path) as doc:
                pages = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap()
                    img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    pages.append(img_data)
                return pages
        else:
            return [Image.open(file_path)]
    except Exception as e:
        print(f"Error visualizing {file_path}: {str(e)}")
        traceback.print_exc()
        return None



