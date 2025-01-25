import os
from dotenv import load_dotenv, find_dotenv
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import numpy as np
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import ChatOpenAI
import fitz

if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_env():
    _ = load_dotenv(find_dotenv())

def check_environment():
    """Check if environment is properly configured"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment variables")
    return True

def process_technical_pdf(pdf_path):
    """Process technical PDFs and drawings with OCR"""
    try:
        print(f"Processing PDF for indexing: {pdf_path}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Convert PDF to images
        print("Converting PDF to images...")
        images = convert_from_path(pdf_path)
        print(f"Converted {len(images)} pages")
        
        extracted_text = []
        
        for i, image in enumerate(images):
            print(f"OCR processing page {i+1}/{len(images)}")
            try:
                # Convert to OpenCV format
                opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Process image
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                denoised = cv2.fastNlMeansDenoising(gray)
                threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Perform OCR
                text = pytesseract.image_to_string(threshold)
                extracted_text.append(f"Page {i + 1}: {text}")
                
            except Exception as page_error:
                print(f"Error processing page {i+1}: {str(page_error)}")
                extracted_text.append(f"Page {i + 1}: [Error: {str(page_error)}]")
        
        # Create document with metadata
        full_text = "\n".join(extracted_text)
        return Document(
            text=full_text,
            metadata={
                "source": pdf_path,
                "pages": len(images),
                "type": "technical_drawing"
            }
        )
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def initialize_llama_index(documents_path="./documents"):
    """Initialize and return LlamaIndex with support for technical documents"""
    try:
        # Load environment
        load_dotenv()
        print("Environment loaded")

        # Configure OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        llm = ChatOpenAI(model="gpt-4", api_key=api_key)
        embed_model = OpenAIEmbedding(api_key=api_key)
        print("OpenAI models configured")

        # Process PDF directly
        try:
            pdf_path = os.path.join(documents_path, "Drawings FLOOR PLANS.pdf")
            print(f"Processing PDF directly: {pdf_path}")
            
            # Process the PDF and create a document
            doc_text = process_technical_pdf(pdf_path)
            if not doc_text:
                raise ValueError("Failed to process PDF")
                
            # Create index from the single document
            print("Creating index...")
            index = VectorStoreIndex.from_documents(
                [doc_text],  # Pass as list
                embed_model=embed_model
            )
            print("Index created successfully")
            return index
            
        except Exception as reader_error:
            print(f"Error in document processing: {str(reader_error)}")
            raise
            
    except Exception as e:
        print(f"Error in initialize_llama_index: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

def create_chat_engine(index):
    """Create a chat engine optimized for technical documents"""
    return index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True,
        system_prompt="""You are an expert in analyzing technical drawings, blueprints, and documentation. 
        Focus on providing precise, technical information from the documents."""
    )

def visualize_document(file_path):
    """Visualize document (PDF or image)"""
    try:
        print(f"Attempting to visualize: {file_path}")
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            print(f"PDF opened successfully, pages: {len(doc)}")
            pages = []
            for i, page in enumerate(doc):
                print(f"Processing page {i+1}")
                pix = page.get_pixmap()
                img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img_data)
            print(f"Successfully processed {len(pages)} pages")
            return pages
        print("Not a PDF file")
        return [Image.open(file_path)]
    except Exception as e:
        print(f"Error visualizing document: {str(e)}")
        return None


