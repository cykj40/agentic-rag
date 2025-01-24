from helper import check_environment, initialize_llama_index, create_chat_engine
import os

def check_documents_folder():
    """Verify documents folder exists and contains supported files"""
    docs_path = "./documents"
    supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg'}
    
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)
        print(f"Created documents folder at {docs_path}")
        print("Please add your technical PDFs and drawings to this folder.")
        return False
    
    files = [f for f in os.listdir(docs_path) if os.path.splitext(f)[1].lower() in supported_extensions]
    if not files:
        print("No supported documents found. Please add PDFs or images to the documents folder.")
        return False
    
    print(f"Found {len(files)} supported documents")
    return True

def main():
    try:
        # Check environment and documents
        check_environment()
        if not check_documents_folder():
            return
        
        print("Loading and processing technical documents...")
        index = initialize_llama_index()
        chat_engine = create_chat_engine(index)
        
        print("\nTechnical Document Analysis Interface")
        print("Type 'exit' to quit")
        print("Type 'help' for example questions")
        print("-" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                break
            
            if user_input.lower() == 'help':
                print("\nExample questions:")
                print("- What are the main dimensions in this blueprint?")
                print("- Can you describe the layout of this drawing?")
                print("- What specifications are listed for [component]?")
                continue
                
            if user_input:
                response = chat_engine.chat(user_input)
                print("\nAssistant:", response.response)
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 