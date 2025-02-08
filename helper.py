import os
import traceback
from dotenv import load_dotenv, find_dotenv
from PIL import Image
import fitz
import pytesseract
import re
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
import easyocr
from shapely.geometry import Polygon, box, LineString
from skimage import feature, morphology, measure
from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\ProgramFiles\Tesseract\tesseract.exe'

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
    
    # Convert numpy types to standard Python types
    processed_rooms = [{
        'area_pixels': int(room['area']),  # Convert numpy.int64 to int
        'perimeter_pixels': float(room['perimeter']),  # Convert numpy.float64 to float
        'centroid': tuple(float(x) for x in room['centroid']),  # Convert numpy array to tuple of floats
        'bbox': tuple(int(x) for x in room['bbox'])  # Convert numpy array to tuple of ints
    } for room in rooms]
    
    return {
        'walls': [{'coords': [(float(x), float(y)) for x, y in wall.coords]} for wall in walls],
        'rooms': processed_rooms,
        'symbols': [{
            'type': symbol['type'],
            'center': tuple(int(x) for x in symbol['center']),
            'radius': int(symbol['radius']) if 'radius' in symbol else None
        } for symbol in symbols],
        'measurements': measurements,
        'page_number': page.number + 1,
        'page_size': (float(page.rect.width), float(page.rect.height))
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


def chunk_text(text: str, max_tokens: int = 1000) -> List[str]:
    """Split text into smaller chunks to stay within token limits."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 chars)
    for word in words:
        word_length = len(word) // 4  # Rough token estimation
        if current_length + word_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def process_pdf_to_document(pdf_path: str) -> List[Document]:
    """Process PDF document and extract text, drawings, and measurements.
    Returns a list of Documents, one per page with proper chunking."""
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
        
        # Join content and chunk it
        full_content = '\n'.join(content)
        content_chunks = chunk_text(full_content, max_tokens=1000)
        
        # Create a document for each chunk
        for chunk_idx, chunk in enumerate(content_chunks):
            chunk_doc = Document(
                text=chunk,
                metadata={
                    'page_number': page.number + 1,
                    'chunk_number': chunk_idx + 1,
                    'total_chunks': len(content_chunks),
                    'filename': os.path.basename(pdf_path),
                    'page_elements': elements
                }
            )
            documents.append(chunk_doc)
    
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


def extract_tables_from_pdf(page: fitz.Page) -> List[pd.DataFrame]:
    """Extract tables from a PDF page and convert them to pandas DataFrames."""
    tables = []
    try:
        # Extract tables using PyMuPDF's built-in table detection
        tab = page.find_tables()
        if tab.tables:
            for idx, table in enumerate(tab.tables):
                # Convert table to a list of lists
                data = [cell.text.strip() for row in table.cells for cell in row]
                rows = table.cells
                if rows:
                    # Create DataFrame
                    df = pd.DataFrame(data)
                    # Try to detect header
                    if len(df) > 1:
                        df.columns = df.iloc[0]
                        df = df[1:]
                    tables.append(df)
    except Exception as e:
        print(f"Error extracting tables: {str(e)}")
    return tables


def extract_technical_content(text: str) -> Dict[str, Any]:
    """Extract technical content like equations, measurements, and specifications."""
    technical_data = {
        'equations': [],
        'measurements': [],
        'specifications': [],
        'references': []
    }
    
    # Extract equations (basic pattern)
    equation_pattern = r'(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*[-+*/\d\s.()]+)|(?:\$[^$]+\$)'
    technical_data['equations'] = re.findall(equation_pattern, text)
    
    # Extract measurements
    measurement_pattern = r'\b\d+(?:\.\d+)?\s*(?:mm|cm|m|kg|g|N|Pa|MPa|°C|°F|Hz|kHz|MHz|W|kW|MW|V|kV|A|mA|Ω|μm|nm)\b'
    technical_data['measurements'] = re.findall(measurement_pattern, text)
    
    # Extract specifications (key-value pairs)
    spec_pattern = r'(?:^|\n)([A-Za-z\s]+):\s*([^:\n]+)'
    technical_data['specifications'] = re.findall(spec_pattern, text)
    
    # Extract references
    ref_pattern = r'\[(\d+)\]|\[([\w\-]+)\]'
    technical_data['references'] = re.findall(ref_pattern, text)
    
    return technical_data


def create_visualization(data: Dict[str, Any], viz_type: str) -> Optional[str]:
    """Create visualizations based on the data type and return as base64 encoded string."""
    try:
        plt.figure(figsize=(10, 6))
        
        if viz_type == 'bar' and isinstance(data, dict):
            plt.bar(list(data.keys()), list(data.values()))
            plt.xticks(rotation=45)
        elif viz_type == 'line' and isinstance(data, dict):
            plt.plot(list(data.keys()), list(data.values()))
            plt.xticks(rotation=45)
        elif viz_type == 'pie' and isinstance(data, dict):
            plt.pie(list(data.values()), labels=list(data.keys()), autopct='%1.1f%%')
        elif viz_type == 'table' and isinstance(data, pd.DataFrame):
            plt.axis('off')
            plt.table(cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')
        
        # Save plot to bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return image_base64
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return None


def extract_financial_content(text: str) -> Dict[str, Any]:
    """Extract financial data like metrics, ratios, and market data."""
    financial_data = {
        'metrics': [],
        'ratios': [],
        'market_data': [],
        'dates': [],
        'currency_amounts': []
    }
    
    # Extract currency amounts and financial figures
    currency_pattern = r'\$?\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|T))?\b'
    financial_data['currency_amounts'] = re.findall(currency_pattern, text)
    
    # Extract financial metrics
    metric_pattern = r'(?:revenue|profit|income|EBITDA|EPS|ROE|ROI|margin|growth)\s*(?:of|:)?\s*' + currency_pattern
    financial_data['metrics'] = re.findall(metric_pattern, text, re.IGNORECASE)
    
    # Extract financial ratios
    ratio_pattern = r'(?:P/E|price[/-]to[/-]earnings|debt[/-]to[/-]equity|current ratio|quick ratio|ROE|ROA)\s*(?:of|:)?\s*\d+\.?\d*'
    financial_data['ratios'] = re.findall(ratio_pattern, text, re.IGNORECASE)
    
    # Extract market data
    market_pattern = r'(?:stock price|market cap|volume|shares outstanding|dividend|yield)\s*(?:of|:)?\s*' + currency_pattern
    financial_data['market_data'] = re.findall(market_pattern, text, re.IGNORECASE)
    
    # Extract dates (for quarterly/annual reports)
    date_pattern = r'\b(?:Q[1-4]\s+\d{4}|FY\s+\d{4}|\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b'
    financial_data['dates'] = re.findall(date_pattern, text)
    
    return financial_data


def create_financial_visualization(data: Dict[str, Any], viz_type: str) -> Optional[str]:
    """Create financial visualizations based on the data type."""
    try:
        plt.style.use('seaborn')  # Use a clean style for financial charts
        plt.figure(figsize=(12, 6))
        
        if viz_type == 'line':
            # Time series data
            plt.plot(list(data.keys()), list(data.values()), marker='o')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.title('Time Series Analysis')
            
        elif viz_type == 'bar':
            # Comparative data
            plt.bar(list(data.keys()), list(data.values()))
            plt.grid(True, axis='y')
            plt.xticks(rotation=45)
            
        elif viz_type == 'pie':
            # Distribution data
            plt.pie(list(data.values()), labels=list(data.keys()), autopct='%1.1f%%')
            plt.title('Distribution Analysis')
            
        elif viz_type == 'candlestick' and isinstance(data, pd.DataFrame):
            # Check if we have OHLC data
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                from mplfinance.original_flavor import candlestick_ohlc
                import matplotlib.dates as mdates
                
                # Convert data for candlestick chart
                quotes = []
                for index, row in data.iterrows():
                    quotes.append((mdates.date2num(index), row['Open'], 
                                 row['High'], row['Low'], row['Close']))
                
                candlestick_ohlc(plt.gca(), quotes, width=0.6, 
                                colorup='g', colordown='r')
                plt.grid(True)
                plt.xticks(rotation=45)
        
        # Add financial chart specific formatting
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        return image_base64
    except Exception as e:
        print(f"Error creating financial visualization: {str(e)}")
        return None


def process_document(file_path: str) -> List[Document]:
    """Process a document and extract text content with financial analysis.
    Returns a list of Document objects with enhanced financial content."""
    documents = []
    
    try:
        if file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    # Extract financial content
                    financial_data = extract_financial_content(text)
                    
                    # Extract tables (potential financial statements)
                    tables = extract_tables_from_pdf(page)
                    
                    # Create enhanced metadata
                    metadata = {
                        'source': file_path,
                        'page_number': page_num + 1,
                        'filename': os.path.basename(file_path),
                        'financial_data': financial_data,
                        'has_tables': len(tables) > 0,
                        'table_count': len(tables),
                        'tables': [df.to_dict() for df in tables] if tables else []
                    }
                    
                    # Create document with enhanced content
                    doc = Document(
                        text=text,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            print(f"Processed {len(documents)} pages from PDF: {os.path.basename(file_path)}")
            
    except Exception as e:
        print(f"Error processing document {file_path}: {str(e)}")
        traceback.print_exc()
    
    return documents


def process_text_content(text: str, source_name: str = "Pasted Text") -> Document:
    """Process pasted text content and extract financial information."""
    try:
        if text.strip():
            # Extract financial content
            financial_data = extract_financial_content(text)
            
            # Create metadata
            metadata = {
                'source': source_name,
                'filename': source_name,
                'financial_data': financial_data,
                'content_type': 'text'
            }
            
            # Create document with the content
            doc = Document(
                text=text,
                metadata=metadata
            )
            
            print(f"Processed text content from: {source_name}")
            return doc
            
    except Exception as e:
        print(f"Error processing text content: {str(e)}")
        traceback.print_exc()
        return None


def initialize_index(documents_path: str = "./documents", text_content: Optional[str] = None, text_source: Optional[str] = None) -> Optional[VectorStoreIndex]:
    """Initialize the financial document Q&A system with both PDF and text content."""
    try:
        # Load environment variables
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
        all_documents = []
        
        # Process PDFs if directory exists
        if os.path.exists(documents_path):
            print(f"Processing documents from: {documents_path}")
            for filename in os.listdir(documents_path):
                file_path = os.path.join(documents_path, filename)
                if filename.lower().endswith('.pdf'):
                    documents = process_document(file_path)
                    all_documents.extend(documents)
        
        # Process pasted text if provided
        if text_content:
            text_doc = process_text_content(text_content, text_source or "Pasted Text")
            if text_doc:
                all_documents.append(text_doc)
        
        if not all_documents:
            raise ValueError("No content was successfully processed")
        
        # Create index
        print("Creating document index...")
        index = VectorStoreIndex.from_documents(all_documents)
        print("Document index created successfully")
        return index
        
    except Exception as e:
        print(f"Index creation error: {str(e)}")
        traceback.print_exc()
        return None


def create_chat_engine(index):
    """Create an enhanced chat engine for financial document analysis."""
    return index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        system_prompt=(
            "You are an expert financial analyst specializing in analyzing financial documents, "
            "earnings reports, and market data. Your capabilities include:\n"
            "1. Analyzing financial statements and metrics\n"
            "2. Interpreting earnings calls and quarterly reports\n"
            "3. Creating financial visualizations and charts\n"
            "4. Calculating key financial ratios and indicators\n"
            "5. Providing market analysis and insights\n\n"
            "When responding:\n"
            "- Present financial data in clear, structured tables\n"
            "- Create visualizations for trend analysis\n"
            "- Highlight key financial metrics and their implications\n"
            "- Compare current vs historical performance\n"
            "- Explain financial terms and calculations\n"
            "- Cite specific sections from financial reports\n"
            "If you're unsure about any financial details, acknowledge the uncertainty "
            "and explain what additional data would be needed for a complete analysis."
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



