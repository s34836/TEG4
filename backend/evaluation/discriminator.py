#def validate_inputs(question: str, context: str, answer: str, api_key: str) -> None:
#def rate_answer_with_llm(question: str, context: str, answer: str, api_key: str) -> Tuple[float, str]:

from typing import Tuple, Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import json

#**************************************************************************
def validate_inputs(question: str, context: str, answer: str, api_key: str) -> None:
    """Validate that all inputs are non-empty strings"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer must be a non-empty string")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("API key must be a non-empty string")

#************************************************************************** 
def rate_answer_with_llm(question: str, context: str, answer: str, api_key: str) -> Tuple[float, str]:

    try:
        validate_inputs(question, context, answer, api_key)
        
        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)

        prompt = PromptTemplate.from_template(
            """Jesteś ekspertem oceniającym odpowiedzi RAG. Oceń odpowiedź na podstawie pytania i kontekstu.

Kryteria oceny:
1. Trafność - czy odpowiedź jest zgodna z pytaniem (0-3 punkty)
2. Dokładność - czy odpowiedź jest oparta na kontekście (0-3 punkty)
3. Kompletność - czy odpowiedź jest pełna i wyczerpująca (0-2 punkty)
4. Zwięzłość - czy odpowiedź jest zwięzła i na temat (0-2 punkty)

Pytanie: {question}

Kontekst: {context}

Odpowiedź: {answer}

Zwróć wynik w formacie JSON:
{{
  "score": <ocena w skali 1-10>,
  "justification": "<krótkie uzasadnienie z odniesieniem do kryteriów>"
}}"""
        )

        chain = prompt | llm
        result = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
        })

        try:
            parsed = json.loads(result.content)
            score_raw = parsed.get("score")

            # 🛡️ defensywne odpakowanie score
            if isinstance(score_raw, list):
                # np. ["7.5"] lub [7.5]
                score_raw = score_raw[0] if score_raw else 0.0

            # 🧼 zamiana tekstu na liczbę (jeśli potrzeba)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                print(f"⚠️ Nieprawidłowy score: {score_raw}")
                score = 0.0

            # 🎯 ograniczenie do 0–10
            score = max(0, min(10, score))
            return score, parsed["justification"]
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {result.content}")
            return 0.0, "Error: Invalid JSON response from LLM"
        except (KeyError, ValueError) as e:
            print(f"Response parsing error: {e}")
            return 0.0, f"Error: Invalid response format - {str(e)}"
            
    except Exception as e:
        print(f"Unexpected error in rate_answer_with_llm: {e}")
        import traceback
        print(traceback.format_exc())
        return 0.0, f"Error: {str(e)}"
