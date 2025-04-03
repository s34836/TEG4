import os
from dotenv import load_dotenv
# Getting API key
dotenv_path = os.path.join('../', 'config', '.env')
load_dotenv(dotenv_path)
api_key = os.getenv('OPENAI_API_KEY_TEG')
if not api_key:
    raise ValueError("OpenAI API key not found. Please set either OPENAI_API_KEY or OPENAI_API_KEY_TEG in your .env file.")
os.environ["OPENAI_API_KEY"] = api_key


import warnings
# Suppress specific LangChain warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import traceback
from typing import List, Dict, Any, Optional
import pandas as pd
from abc import ABC, abstractmethod
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import numpy as np

from backend.comparison.comparison_engine import generate_all_variants
from backend.evaluation.discriminator import rate_answer_with_llm
from backend.comparison.prompt_techniques import (
    context_aware_prompt,
    role_prompt,
    chain_of_thought_prompt
)
#**************************************************************************

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_correctness
)
#from ragas.schema import Record
from ragas import evaluate

from datasets import Dataset
from ragas import evaluate as ragas_evaluate  # avoid name clash
#**************************************************************************

def to_float_safe(value):
    """Bezpieczna konwersja do float – wypłaszcza listy i obsługuje błędy."""
    if isinstance(value, list):
        value = value[0] if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"⚠️ Nieprawidłowa wartość metryki: {value}")
        return 0.0


#**************************************************************************
class RAGEvaluator(ABC):
    """Abstract base class for RAG evaluation"""
    
    @abstractmethod
    def evaluate(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """Evaluate a single RAG output"""
        pass

class RAGAsMetricsEvaluator(RAGEvaluator):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key
        )
    
    def evaluate(self, question: str, answer: str, context: str, ground_truth: str) -> Dict[str, float]:
        try:
            # Prepare data for RAGAS evaluation
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [[context]],
                "reference": [ground_truth]  # Add
            }
            dataset = Dataset.from_dict(data)
            
            # Calculate RAGAS metrics
            result = ragas_evaluate(
                dataset=dataset,
                metrics=[
                    answer_relevancy,
                    faithfulness,
                    answer_correctness
                ]
            )
            
            # Get LLM-based evaluation for comparison
            llm_score, llm_justification = rate_answer_with_llm(
                question=question,
                context=context,
                answer=answer,
                api_key=self.api_key
            )
         
            return {
                "answer_relevancy": to_float_safe(result["answer_relevancy"]),
                "faithfulness": to_float_safe(result["faithfulness"]),
                "answer_correctness": to_float_safe(result["answer_correctness"]),
                "llm_score": llm_score / 10.0,
                "llm_justification": llm_justification
            }
        except Exception as e:
            print(f"Error in RAGAS evaluation: {e}")
            return {
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,                
                "answer_correctness": 0.0,
                "llm_score": 0.0,
                "llm_justification": f"Error: {str(e)}"
            }

class RAGEvaluationPipeline:
    """Main class for running RAG evaluations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.evaluator = RAGAsMetricsEvaluator(api_key)
    
    def run_evaluation(self, question : str, ground_truth : str) -> pd.DataFrame:
        """Run the main evaluation pipeline"""
        try:

            
            # Generate variants for each question
            results = generate_all_variants(question)
            
            rows = []
            for question, variants in results.items():
                print(f"\nProcessing question: '{question}'")
                for variant_data in variants:
                    variant = variant_data.get("variant", "Unknown")
                    answer = variant_data.get("output", "No output generated")
                    context = variant_data.get("retrieved_context", "No context retrieved")
                    print(f"  Evaluating variant: {variant}")
                    
                    # Get evaluation metrics for this specific case
                    metrics = self.evaluator.evaluate(
                        question=question,
                        answer=answer,
                        context=context,
                        ground_truth=ground_truth
                    )
                    
                    # Combine data into a row
                    row_data = {
                        "question": question,
                        "variant": variant,
                        "answer": answer,
                        "context": context,
                        "ground_truth": ground_truth,
                        **metrics
                    }
                    
                    rows.append(row_data)

            df = pd.DataFrame(rows)
            return df

        except Exception as e:
            print(f"Error in evaluation pipeline: {e}")
            print(traceback.format_exc())
            raise       
      
def main():
    """Main function to run the evaluation"""
    try:
        # Load .env file from the project root
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
        
        # Run the evaluation pipeline
        pipeline = RAGEvaluationPipeline(api_key)
        
        # Define 3 specific questions for evaluation
        questions = [
            "Co to jest motywacja?",
            "Na czym polega system premiowania nauczycieli?",
            "Jakie są metody oceny pracy nauczyciela?"
        ]

        ground_truths = {
            "Co to jest motywacja?": "Motywacja to wewnętrzny lub zewnętrzny impuls pobudzający człowieka do działania i wytrwałości w dążeniu do określonego celu.",
            "Na czym polega system premiowania nauczycieli?": "System nagradzania nauczycieli polega na przyznawaniu wyróżnień, awansów zawodowych, nagród finansowych lub rzeczowych w uznaniu za osiągnięcia dydaktyczne, wychowawcze i zaangażowanie w rozwój szkoły.",
            "Jakie są metody oceny pracy nauczyciela?": "Metody oceny pracy nauczyciela obejmują obserwację lekcji, analizę dokumentacji, ankiety uczniowskie i rodzicielskie, samoocenę oraz wyniki uczniów."
        }            
        print(f"Evaluating RAG for {len(questions)} questions...")   
        individual_dfs = []

        for question in questions:
            ground_truth = ground_truths[question]
            print(f"Running evaluation for: {question}")
            df = pipeline.run_evaluation(question=question, ground_truth=ground_truth)
            individual_dfs.append(df)

        # Combine all results into a single DataFrame
        results_df = pd.concat(individual_dfs, ignore_index=True)
        
        # Create evaluation directory if it doesn't exist
        os.makedirs("backend/evaluation", exist_ok=True)
        
        # Clean the data by replacing newlines with spaces
        for column in ['answer', 'context', 'llm_justification']:
            if column in results_df.columns:
                results_df[column] = results_df[column].apply(lambda x: ' '.join(str(x).splitlines()))
        
        # Rename columns to include spaces
        column_mapping = {
            'question': 'Question',
            'variant': 'Variant',
            'answer': 'Answer',
            'context': 'Context',
            'answer_relevancy': 'Answer Relevancy',            
            'faithfulness': 'Faithfulness',
            'answer_correctness': 'Answer Correctness',        
            'llm_score': 'LLM Score',
            'llm_justification': 'LLM Justification'
        }
        results_df = results_df.rename(columns=column_mapping)
        
        # Save to CSV with tab separator and proper quoting
        csv_path = os.path.join("backend/evaluation", "evaluation_results.csv")
        results_df.to_csv(csv_path, index=False, sep='\t', quoting=1, escapechar='\\')
        print(f"\nResults saved to: {csv_path}")
        
        # Print first few rows to verify the output
        print("\nFirst few rows of the saved data:")
        print(results_df.head())
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
