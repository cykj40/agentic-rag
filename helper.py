import os
import traceback
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import fitz
import pytesseract
import re
from typing import Dict, List, Tuple
import cv2
import numpy as np
import easyocr
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def extract_measurements(text: str) -> List[Dict[str, str]]:
    """Extract measurements with their context from text."""
    # Common measurement patterns in blueprints
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|ft|\'|\"|\binch(?:es)?\b|\bfeet\b)',  # Basic measurements
        r'(\d+(?:\.\d+)?)\s*(?:x|\*)\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|ft|\'|\")',  # Dimensions like 10x20 ft
        r'(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft|square\s*feet)',  # Area measurements
        r'(?:width|length|height|depth|radius|diameter)\s*(?:of|:|\=)?\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|ft|\'|\")',  # Labeled measurements
    ]
    
    measurements = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get some context around the measurement
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()
            
            measurements.append({
                "measurement": match.group(0),
                "context": context,
                "location": f"chars {match.start()}-{match.end()}"
            })
    
    return measurements


def detect_shapes(image_np: np.ndarray) -> List[Dict]:
    """Detect geometric shapes in blueprint using OpenCV."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Get shape type based on number of vertices
        vertices = len(approx)
        shape_type = 'unknown'
        if vertices == 3:
            shape_type = 'triangle'
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.95 <= aspect_ratio <= 1.05:
                shape_type = 'square'
            else:
                shape_type = 'rectangle'
        elif vertices > 4:
            shape_type = 'circle' if vertices > 8 else 'polygon'
        
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        shapes.append({
            'type': shape_type,
            'vertices': vertices,
            'area': area,
            'perimeter': perimeter,
            'coordinates': approx.tolist()
        })
    
    return shapes


def detect_text_regions(image_np: np.ndarray) -> List[Dict]:
    """Extract text regions using Tesseract and EasyOCR."""
    # Get text regions from EasyOCR
    results = reader.readtext(image_np)
    
    text_regions = []
    for (bbox, text, conf) in results:
        # Convert bbox to Shapely polygon for geometric operations
        polygon = Polygon([
            (bbox[0][0], bbox[0][1]),  # top-left
            (bbox[1][0], bbox[1][1]),  # top-right
            (bbox[2][0], bbox[2][1]),  # bottom-right
            (bbox[3][0], bbox[3][1])   # bottom-left
        ])
        
        text_regions.append({
            'text': text,
            'confidence': conf,
            'bbox': bbox,
            'area': polygon.area,
            'centroid': (polygon.centroid.x, polygon.centroid.y)
        })
    
    return text_regions


def analyze_blueprint_elements(page: fitz.Page) -> Dict:
    """Analyze different elements in a blueprint page."""
    # Convert page to image
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    # Detect shapes
    shapes = detect_shapes(img)
    
    # Detect text regions
    text_regions = detect_text_regions(img)
    
    # Extract measurements from text
    measurements = []
    for region in text_regions:
        text = region['text']
        # Look for common measurement patterns
        if any(unit in text.lower() for unit in ['mm', 'cm', 'm', 'ft', 'in', '"', "'"]):
            measurements.append({
                'text': text,
                'location': region['centroid'],
                'confidence': region['confidence']
            })
    
    return {
        'shapes': shapes,
        'text_regions': text_regions,
        'measurements': measurements,
        'page_number': page.number + 1,
        'page_size': (page.rect.width, page.rect.height)
    }


def process_pdf_to_document(pdf_path: str) -> Document:
    """Process PDF document and extract text, drawings, and measurements."""
    doc = fitz.open(pdf_path)
    content = []
    metadata = {
        'total_pages': len(doc),
        'filename': os.path.basename(pdf_path),
        'elements_by_page': []
    }
    
    for page in doc:
        # Extract text
        text = page.get_text()
        
        # Analyze blueprint elements
        elements = analyze_blueprint_elements(page)
        metadata['elements_by_page'].append(elements)
        
        # Add page content with context
        content.append(f"Page {page.number + 1}:\n{text}\n")
        
        # Add shape descriptions
        for shape in elements['shapes']:
            content.append(f"Found {shape['type']} with area {shape['area']:.2f}\n")
        
        # Add measurement descriptions
        for measurement in elements['measurements']:
            content.append(f"Measurement found: {measurement['text']}\n")
    
    return Document(text='\n'.join(content), metadata=metadata)


def ocr_pdf_document(pdf_path: str) -> Document:
    """
    Perform OCR on each page of a scanned PDF and return a Document with the extracted text.
    Requires Tesseract + pytesseract installed.
    """
    print(f"Attempting OCR on PDF: {pdf_path}")
    try:
        with fitz.open(pdf_path) as doc:
            extracted_pages = []
            for page_index in range(len(doc)):
                page = doc[page_index]
                # Convert page to an image
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Run Tesseract OCR on the image
                ocr_text = pytesseract.image_to_string(img)
                extracted_pages.append(f"Page {page_index + 1}:\n{ocr_text.strip()}")
            
            combined_text = "\n\n".join(extracted_pages).strip()
            
            if not combined_text:
                # If OCR yields nothing, return None
                return None
            
            return Document(
                text=combined_text,
                metadata={
                    "source": pdf_path,
                    "type": "raster_blueprint",
                    "filename": os.path.basename(pdf_path)
                }
            )
    except Exception as e:
        print(f"Error in ocr_pdf_document: {e}")
        traceback.print_exc()
        return None


def initialize_llama_index(documents_path: str = "./documents"):
    """Initialize RAG system with technical document understanding,
    handling both text-based and scanned PDFs."""
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
                
                # First, try text-based extraction
                doc = process_pdf_to_document(file_path)
                
                # If doc is None or doc.text is empty, fallback to OCR
                if (not doc) or (not doc.text.strip()):
                    print(f"No text found in {filename}, switching to OCR.")
                    doc = ocr_pdf_document(file_path)
                
                if doc:
                    documents.append(doc)
                    print(f"Successfully processed (PDF) with text or OCR: {filename}")
        
        if not documents:
            raise ValueError("No documents were successfully processed")
        
        # Create index with larger chunk size
        print("Creating technical document index...")
        from llama_index.core.node_parser import SimpleNodeParser
        
        # Configure parser with larger chunk size
        parser = SimpleNodeParser.from_defaults(
            chunk_size=4096,  # Increased from default 1024
            chunk_overlap=50
        )
        
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[parser]  # Use our configured parser
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
        chat_mode="context",
        verbose=True,
        context_template=(
            "You are an expert in analyzing technical drawings, blueprints, and engineering documentation. "
            "Focus on providing precise, technical information including measurements, dimensions, and specifications. "
            "\n\nWhen discussing measurements:\n"
            "- Always specify the units (ft, inches, mm, etc.)\n"
            "- Provide context for where the measurement was found\n"
            "- If relevant, explain what the measurement refers to (e.g., room width, door height)\n"
            "- If asked about areas or volumes, show your calculations\n"
            "\n\nWhen discussing blueprints:\n"
            "- Reference specific pages and sections\n"
            "- Describe the location of elements using clear spatial terms\n"
            "- Mention any relevant annotations or notes\n"
            "- If discussing layout, provide clear directional guidance\n"
            "\n\nIf you're unsure about any technical detail, say so rather than making assumptions. "
            "Use technical terminology appropriate for architectural and engineering contexts.\n\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given this context, please respond to the question: {query_str}\n"
        )
    )


def visualize_document(file_path):
    """Convert each page of the PDF into an image for display in Streamlit."""
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
            # If an image file, just load it
            return [Image.open(file_path)]
    except Exception as e:
        print(f"Error visualizing {file_path}: {str(e)}")
        traceback.print_exc()
        return None



