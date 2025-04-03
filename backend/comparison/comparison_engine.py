#def get_retriever_config():
#def safe_llm_call(llm, input_text: str) -> str:
#def generate_all_variants(questions: List[str]) -> Dict[str, List[Dict[str, str]]]:

from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from backend.comparison.prompt_techniques import context_aware_prompt, role_prompt, chain_of_thought_prompt
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load environment variables, paths and othe globals
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"current_dir: {current_dir}")
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
print(f"project_root: {project_root}")
dotenv_path = os.path.join(project_root, 'config', '.env')
print(f"dotenv_path: {dotenv_path}")
index_dir = os.path.join(current_dir, "../faiss_index")
print(f"index_dir: {index_dir}")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


import re

def clean_text(text: str) -> str:
    # 1. Zamiana niedrukowalnych znaków i typografii
    text = text.replace('\xa0', ' ')  # non-breaking space
    text = re.sub(r'[\u2013\u2014\u2015]', '-', text)  # różne myślniki
    text = text.replace('\u2026', '...')  # wielokropek
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # cudzysłowy
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # apostrofy

    # 2. Scal złamane słowa z myślnikami (np. "auto-\nnomia" → "autonomia")
    text = re.sub(r'(\w+)[\-‐‑‒–—−]\n(\w+)', r'\1\2', text)

    # 3. Scal łamanie linii w środku zdań (pozostaw akapity z \n\n)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # 4. Zamiana błędnych znaków diakrytycznych na poprawne litery
    diacritic_map = {
        'ą': 'ą', 'a̧': 'ą', 'ą̧': 'ą',
        'ć': 'ć',
        'ę': 'ę',
        'l̦': 'ł', 'l~': 'ł',
        'ń': 'ń',
        'ó': 'ó',
        'ś': 'ś',
        'ź': 'ź',
        'ż': 'ż',
    }
    for wrong, correct in diacritic_map.items():
        text = text.replace(wrong, correct)

    # 5. Usuwanie artefaktów OCR (zbędne plusy, nawiasy)
    text = re.sub(r'[+]+', '', text)
    text = re.sub(r'[)]{2,}', ')', text)

    return text.strip()

# Try to load from multiple possible locations
env_paths = [
    dotenv_path,
    os.path.join(project_root, '.env'),
    os.path.join(os.path.dirname(project_root), 'config', '.env'),
    os.path.join(os.path.dirname(project_root), '.env')
]

env_loaded = False
for path in env_paths:
    if os.path.exists(path):
        load_dotenv(path)
        env_loaded = True
        print(f"Loaded environment variables from: {path}")
        break

if not env_loaded:
    print("Warning: Could not find .env file in any of these locations:")
    for path in env_paths:
        print(f"- {path}")

api_key = os.getenv('OPENAI_API_KEY_TEG')
if not api_key:
    raise ValueError("OPENAI_API_KEY_TEG not found in environment variables. Please ensure it is set in your .env file.")


#**************************************************************************
def get_retriever_config():
    """Get the default retriever configuration"""
    return {
        "search_type": "similarity",
        "search_kwargs": {"k": 4},  # Number of documents to retrieve
        "score_threshold": 0.5  # Minimum similarity score
    }


#**************************************************************************
def safe_llm_call(llm, input_text: str) -> str:
    """Safely make LLM calls with error handling"""
    try:
        return llm.predict(input_text)
    except Exception as e:
        print(f"Error in LLM call: {e}")
        return f"Error generating response: {str(e)}"

#**************************************************************************
def generate_all_variants(question: str) -> Dict[str, List[Dict[str, str]]]:
    """Generate answers using different RAG variants and prompt techniques for a single question."""
    try:
        print(f"Loading FAISS index from: {index_dir}")
        db = FAISS.load_local(index_dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(**get_retriever_config())
        llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo", temperature=0)

        source_docs = retriever.get_relevant_documents("")  # Get all documents
        outputs = []
        try:
            # 3. Choose 3 pairs of additional modules that can improve our RAG
            # 1. Baseline
            docs = retriever.get_relevant_documents(question)
            baseline_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            response = baseline_chain.run(question)
            outputs.append({
                "variant": "Module: Baseline", 
                "output": response,
                #"retrieved_context": "\n\n".join(doc.page_content for doc in docs)
                "retrieved_context": "\n\n".join(clean_text(doc.page_content) for doc in docs)
            })

            # 2. ParentRetriever
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            parent_vectorstore = FAISS.from_documents(source_docs, embedding_model)
            parent_retriever = ParentDocumentRetriever(
                vectorstore=parent_vectorstore,
                docstore=InMemoryStore(),
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
            parent_retriever.add_documents(source_docs)
            parent_docs = parent_retriever.get_relevant_documents(question)
            parent_chain = RetrievalQA.from_chain_type(llm=llm, retriever=parent_retriever)
            parent_response = parent_chain.run(question)
            parent_context = "\n\n".join(doc.page_content for doc in parent_docs) if parent_docs else "No context retrieved"
            outputs.append({
                "variant": "Module: ParentRetriever", 
                "output": parent_response,
                #"retrieved_context": parent_context
                "retrieved_context": clean_text(parent_context)
            })

            # 3. MultiQuery
            multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
            multi_docs = multi_retriever.get_relevant_documents(question)
            multi_chain = RetrievalQA.from_chain_type(llm=llm, retriever=multi_retriever)
            multi_response = multi_chain.run(question)
            outputs.append({
                "variant": "Module: MultiQuery", 
                "output": multi_response,
                #"retrieved_context": "\n\n".join(doc.page_content for doc in multi_docs)
                "retrieved_context": clean_text("\n\n".join(doc.page_content for doc in multi_docs))
            })

            #4. Choose 3 different Prompt Engineering Techniques that can improve our RAG
            prompt_docs = retriever.get_relevant_documents(question)
            joined_docs = "\n\n".join(doc.page_content for doc in prompt_docs)

            context_aware_input = context_aware_prompt(question, joined_docs)
            outputs.append({
                "variant": "Prompt: ContextAwarePrompt", 
                "output": safe_llm_call(llm, context_aware_input),
                #"retrieved_context": joined_docs
                "retrieved_context": clean_text(joined_docs)
            })

            role_input = role_prompt(question, joined_docs)
            outputs.append({
                "variant": "Prompt: RolePrompt", 
                "output": safe_llm_call(llm, role_input),
                #"retrieved_context": joined_docs
                "retrieved_context": clean_text(joined_docs)
            })

            cot_input = chain_of_thought_prompt(question, joined_docs)
            outputs.append({
                "variant": "Prompt: ChainOfThought", 
                "output": safe_llm_call(llm, cot_input),
                #"retrieved_context": joined_docs
                "retrieved_context": clean_text(joined_docs) 
            })

        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            outputs.append({
                "variant": "Error",
                "output": f"Error processing question: {str(e)}",
                "retrieved_context": ""
            })

        return {question: outputs}

    except Exception as e:
        print(f"Critical error in generate_all_variants: {e}")
        import traceback
        print(traceback.format_exc())
        return {}

#**************************************************************************