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
from shapely.geometry import Polygon, box, LineString
from skimage import feature, morphology, measure
from scipy import ndimage
import matplotlib.pyplot as plt

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


def detect_walls(image_np: np.ndarray) -> List[LineString]:
    """Detect walls in blueprint using edge detection and line detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = feature.canny(gray, sigma=2)
    
    # Dilate edges to connect nearby lines
    dilated = morphology.dilation(edges, morphology.disk(2))
    
    # Detect lines using probabilistic Hough transform
    lines = cv2.HoughLinesP(
        dilated.astype(np.uint8) * 255,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10
    )
    
    wall_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            wall_lines.append(LineString([(x1, y1), (x2, y2)]))
    
    return wall_lines


def detect_rooms(image_np: np.ndarray) -> List[Dict]:
    """Detect rooms and calculate their areas."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Remove small objects and fill holes
    cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=1000)
    filled = ndimage.binary_fill_holes(cleaned)
    
    # Label connected components (rooms)
    labeled_rooms = measure.label(filled)
    regions = measure.regionprops(labeled_rooms)
    
    rooms = []
    for region in regions:
        # Get room boundary
        boundary = region.coords
        # Convert to polygon
        polygon = Polygon([(point[1], point[0]) for point in boundary])
        
        rooms.append({
            'area': region.area,
            'perimeter': region.perimeter,
            'centroid': region.centroid,
            'bbox': region.bbox,
            'polygon': polygon
        })
    
    return rooms


def detect_symbols(image_np: np.ndarray) -> List[Dict]:
    """Detect architectural symbols (doors, windows, etc.)."""
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Template matching could be used here for specific symbols
    # For now, we'll detect basic shapes
    
    # Detect circles (could be electrical outlets, columns)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=30
    )
    
    symbols = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            symbols.append({
                'type': 'circle',
                'center': (x, y),
                'radius': r
            })
    
    return symbols


def analyze_blueprint_elements(page: fitz.Page) -> Dict:
    """Analyze different elements in a blueprint page."""
    # Convert page to image
    pix = page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    # Detect walls
    walls = detect_walls(img)
    
    # Detect rooms
    rooms = detect_rooms(img)
    
    # Detect architectural symbols
    symbols = detect_symbols(img)
    
    # Extract text and measurements from page
    text = page.get_text()
    measurements = extract_measurements(text)
    
    return {
        'walls': [{'coords': list(wall.coords)} for wall in walls],
        'rooms': [{
            'area_pixels': room['area'],
            'perimeter_pixels': room['perimeter'],
            'centroid': room['centroid'],
            'bbox': room['bbox']
        } for room in rooms],
        'symbols': symbols,
        'measurements': measurements,
        'page_number': page.number + 1,
        'page_size': (page.rect.width, page.rect.height)
    }


def extract_measurements(text: str) -> List[Dict[str, str]]:
    """Extract measurements with their context from text."""
    # Common measurement patterns in blueprints
    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:mm|cm|m|ft|\'|\"|\binch(?:es)?\b|\bfeet\b)',  # Basic measurements
        r'(\d+(?:\.\d+)?)\s*(?:x|\*)\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|ft|\'|\")',  # Dimensions
        r'(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft|square\s*feet)',  # Area measurements
        r'(?:width|length|height|depth|radius|diameter)\s*(?:of|:|\=)?\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|m|ft|\'|\")'  # Labeled measurements
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


def process_pdf_to_document(pdf_path: str) -> List[Document]:
    """Process PDF document and extract text, drawings, and measurements.
    Returns a list of Documents, one per page."""
    doc = fitz.open(pdf_path)
    documents = []
    
    for page in doc:
        # Analyze blueprint elements
        elements = analyze_blueprint_elements(page)
        
        # Create content for this page
        content = []
        content.append(f"Page {page.number + 1}:\n")
        
        # Add room descriptions with detailed measurements
        for i, room in enumerate(elements['rooms']):
            content.append(f"Room {i+1}:\n")
            content.append(f"  Area: {room['area_pixels']} pixels\n")
            content.append(f"  Perimeter: {room['perimeter_pixels']} pixels\n")
            content.append(f"  Location: center at {room['centroid']}\n")
            content.append(f"  Bounding box: {room['bbox']}\n")
        
        # Add measurement descriptions
        for measurement in elements['measurements']:
            content.append(f"Measurement: {measurement['measurement']}\n")
            content.append(f"Context: {measurement['context']}\n")
            content.append(f"Location: {measurement['location']}\n")
        
        # Add symbol descriptions
        for symbol in elements['symbols']:
            content.append(f"Symbol: {symbol['type']} at {symbol['center']}\n")
        
        # Create a document for this page
        page_doc = Document(
            text='\n'.join(content),
            metadata={
                'page_number': page.number + 1,
                'filename': os.path.basename(pdf_path),
                'page_elements': elements
            }
        )
        documents.append(page_doc)
    
    return documents


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
        all_documents = []
        
        for filename in os.listdir(documents_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(documents_path, filename)
                
                # Process PDF into per-page documents
                page_documents = process_pdf_to_document(file_path)
                if page_documents:
                    all_documents.extend(page_documents)
                    print(f"Successfully processed {len(page_documents)} pages from: {filename}")
        
        if not all_documents:
            raise ValueError("No documents were successfully processed")
        
        # Create index with standard chunk size (now safe because metadata is per-page)
        print("Creating technical document index...")
        from llama_index.core.node_parser import SimpleNodeParser
        
        parser = SimpleNodeParser.from_defaults(
            chunk_size=4096,
            chunk_overlap=50
        )
        
        index = VectorStoreIndex.from_documents(
            all_documents,
            transformations=[parser]
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
        system_prompt=(
            "You are an expert in analyzing technical drawings and blueprints. "
            "Focus on providing precise information about:\n"
            "1. Room layouts and dimensions\n"
            "2. Wall locations and lengths\n"
            "3. Architectural symbols and their meanings\n"
            "4. Measurements and scale information\n"
            "\nWhen discussing measurements:\n"
            "- Convert pixel measurements to real units when scale information is available\n"
            "- Provide context about where elements are located\n"
            "- Explain relationships between different spaces\n"
            "\nIf you're unsure about any measurements or interpretations, say so."
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



