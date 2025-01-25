import os
import traceback
from dotenv import load_dotenv, find_dotenv
from pdf2image import convert_from_path
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
    """
    Convert a single PDF to a LlamaIndex Document via OCR.
    Each page is turned into text, then joined into a single string.
    """
    print(f"Processing PDF for OCR: {pdf_path}")
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    print(f"  -> Found {len(images)} pages in PDF")

    extracted_texts = []
    for i, image in enumerate(images):
        print(f"  -> OCR page {i+1}/{len(images)} of {os.path.basename(pdf_path)}")
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Preprocess for better OCR
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Perform OCR
        text = pytesseract.image_to_string(threshold)
        extracted_texts.append(f"Page {i+1}:\n{text.strip()}")

    full_text = "\n\n".join(extracted_texts)
    # Create a single Document object with the text from all pages
    return Document(
        text=full_text,
        metadata={
            "source": pdf_path,
            "type": "technical_drawing"
        },
    )


def build_documents_list(docs_folder: str) -> list:
    """
    Iterate over all PDFs in the docs_folder, OCR each one,
    and return a list of LlamaIndex Document objects.
    """
    all_docs = []
    for fname in os.listdir(docs_folder):
        fpath = os.path.join(docs_folder, fname)
        if fpath.lower().endswith('.pdf'):
            print(f"Found PDF: {fname}, converting to Document...")
            try:
                doc = process_pdf_to_document(fpath)
                all_docs.append(doc)
            except Exception as e:
                print(f"Error processing {fname}: {str(e)}")
                traceback.print_exc()
    return all_docs


def initialize_llama_index(documents_path: str = "./documents"):
    """
    Initialize and return a VectorStoreIndex with all PDFs in `documents_path`.
    """
    try:
        # Load environment
        load_env()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        print("Building documents from all PDFs in:", documents_path)
        docs = build_documents_list(documents_path)
        if not docs:
            print("No documents were successfully processed via OCR.")
            return None
        
        print("Configuring LLM and embeddings...")
        llm = ChatOpenAI(model="gpt-4", api_key=api_key)
        embed_model = OpenAIEmbedding(api_key=api_key)
        
        print("Creating index...")
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        print("Index created successfully.")
        return index
    except Exception as e:
        print(f"Error in initialize_llama_index: {str(e)}")
        traceback.print_exc()
        return None


def create_chat_engine(index):
    """
    Create a chat engine optimized for technical (blueprint) documents.
    """
    return index.as_chat_engine(
        chat_mode="condense_question",
        verbose=True,
        system_prompt=(
            "You are an expert in analyzing technical drawings, blueprints, "
            "and documentation. Focus on providing precise, technical "
            "information from the documents."
        ),
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



