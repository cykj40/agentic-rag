# Add your utilities or helper functions to this file.

import os
from dotenv import load_dotenv, find_dotenv
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import cv2
import numpy as np
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import fitz

# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService                                                                                                                                     
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

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
        print(f"Processing PDF: {pdf_path}")
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        extracted_text = []
        
        total_pages = len(images)
        print(f"Found {total_pages} pages")
        
        # Process all pages
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{total_pages}")
            
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Image preprocessing for better OCR
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Perform OCR
            text = pytesseract.image_to_string(threshold)
            extracted_text.append(f"Page {i+1}: {text}")
        
        result = "\n".join(extracted_text)
        print(f"Successfully processed {total_pages} pages")
        return result
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise  # Re-raise the exception for proper error handling

def initialize_llama_index(documents_path="./documents"):
    """Initialize and return LlamaIndex with support for technical documents"""
    # Load environment variables
    load_dotenv()
    
    # Configure OpenAI
    llm = OpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Update global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Configure document reader with custom PDF handling
    reader = SimpleDirectoryReader(
        documents_path,
        file_extractor={
            ".pdf": lambda x: process_technical_pdf(x)
        }
    )
    
    try:
        documents = reader.load_data()
        if not documents:
            print("No documents were loaded successfully.")
            return None
            
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model
        )
        return index
    except Exception as e:
        print(f"Error initializing index: {str(e)}")
        return None

def create_chat_engine(index):
    """Create a chat engine optimized for technical documents"""
    chat_engine = index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True,
        system_prompt="""You are an expert in analyzing technical drawings, blueprints, and documentation. 
        Focus on providing precise, technical information from the documents while maintaining accuracy. 
        If you encounter measurements, specifications, or technical details, state them explicitly."""
    )
    return chat_engine

def visualize_document(file_path):
    """Visualize document (PDF or image)"""
    try:
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap()
                img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages.append(img_data)
            return pages  # Return list of all pages
        else:
            return [Image.open(file_path)]  # Return single image in list for consistency
    except Exception as e:
        print(f"Error visualizing document: {str(e)}")
        return None
