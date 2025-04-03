import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import fitz  # PyMuPDF
import re

# Get the absolute path to the data directory
current_dir = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(current_dir, "data", "book2.pdf")
# Define backend directory path
BACKEND_DIR = os.path.join(current_dir, "backend")

def fix_polish_chars(text):
    """
    Naprawia błędnie rozpoznane polskie znaki lub artefakty w tekście PDF.
    Dodaj kolejne wzorce jeśli potrzeba.
    """
    replacements = {
        '%&': 'ę',
        '%': 'ę',
        '&': 'ą',
        '$': 'ś',
        '#': 'ł',
        '@': 'ż',
        '(': 'ź',
        '`': 'ć',
        '¥': 'ó',
        '¡': 'ń',
        '¤': 'Ś',
        '¢': 'Ł',
        '§': 'Ź',
        '¨': 'Ż',
        '¬': 'Ć',
        '£': 'Ń'
    }

    for broken, correct in replacements.items():
        text = text.replace(broken, correct)

    # Usuń przypadkowe znaki specjalne, które mogły zostać
    text = re.sub(r'[^\x00-\x7FąćęłńóśźżĄĆĘŁŃÓŚŹŻ\w\s.,:;!?\"\'()\-\n]', '', text)
    return text


def main():
    try:
        # Check if PDF file exists
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"PDF file not found at: {PDF_PATH}")

        # Check if backend directory exists
        if not os.path.exists(BACKEND_DIR):
            raise FileNotFoundError(f"Backend directory not found at: {BACKEND_DIR}")

        # Wczytaj PDF za pomocą fitz
        doc = fitz.open(PDF_PATH)
        print(f"Loaded {len(doc)} pages from the PDF")

        full_text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text:
                full_text += fix_polish_chars(page_text) + "\n"

        if not full_text.strip():
            raise ValueError("No text was extracted from the PDF")

        print("Sample text from first page:", full_text[:300])

        # Podział na chunki 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased from 500
            chunk_overlap=100,  # Increased from 50
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]  # Added "." for better sentence splitting
        )
        chunks = text_splitter.split_text(full_text)
        
        if not chunks:
            raise ValueError("No chunks were created from the text")
            
        print(f"Split document into {len(chunks)} chunks")
        print("Sample chunk:", chunks[0][:300])

        # Zamień na dokumenty LangChain
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Użyj lokalnego modelu
        print("Using local HuggingFace embedding model")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Zbuduj FAISS
        vectorstore = FAISS.from_documents(documents, embedding_model)
        print("Vector database created successfully")

        # Create the faiss_index directory in backend if it doesn't exist
        index_dir = os.path.join(BACKEND_DIR, "faiss_index")
        os.makedirs(index_dir, exist_ok=True)

        # Zapisz
        vectorstore.save_local(index_dir)
        print(f"Vector database saved locally at: {index_dir}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the PDF file and backend directory exist in the correct location")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check the PDF file content")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        if 'doc' in locals():
            doc.close()


if __name__ == "__main__":
    main()