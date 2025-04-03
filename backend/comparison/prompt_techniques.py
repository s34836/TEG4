# def validate_inputs(question: str, context: str) -> None:
# def context_aware_prompt(question: str, context: str) -> str:
# def role_prompt(question: str, context: str) -> str:
# def chain_of_thought_prompt(question: str, context: str) -> str:
from typing import Optional

#**************************************************************************
def validate_inputs(question: str, context: str) -> None:
    """Validate that inputs are not empty and are strings"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("Context must be a non-empty string")

#**************************************************************************
def context_aware_prompt(question: str, context: str) -> str:
    """
    Generate a context-aware prompt that explicitly instructs the model to use the provided context.
    """
    validate_inputs(question, context)
    return f"""Przeanalizuj uważnie poniższy kontekst i odpowiedz na pytanie. Używaj TYLKO informacji z kontekstu.
Jeśli odpowiedź nie znajduje się w kontekście, napisz "Nie mogę odpowiedzieć na to pytanie na podstawie dostępnego kontekstu."

Kontekst:
{context}

Pytanie:
{question}

Odpowiedź (bazując wyłącznie na powyższym kontekście):"""

#**************************************************************************
def role_prompt(question: str, context: str) -> str:
    """
    Generate a role-based prompt that frames the response from a teacher's perspective.
    """
    validate_inputs(question, context)
    return f"""Wciel się w rolę doświadczonego nauczyciela akademickiego, specjalizującego się w tej dziedzinie.
Twoim zadaniem jest udzielenie jasnej, dokładnej i edukacyjnej odpowiedzi na pytanie studenta.
Bazuj WYŁĄCZNIE na poniższych materiałach źródłowych.

Materiały źródłowe:
{context}

Pytanie studenta:
{question}

Odpowiedź nauczyciela (używając tylko powyższych materiałów):
1. Najpierw krótko wyjaśnię kluczowe pojęcia
2. Następnie odpowiem na pytanie
3. Na koniec podsumuję najważniejsze punkty

"""
#**************************************************************************
def chain_of_thought_prompt(question: str, context: str) -> str:
    """
    Generate a chain-of-thought prompt that encourages step-by-step reasoning.
    """
    validate_inputs(question, context)
    return f"""Przeanalizuj poniższy kontekst i odpowiedz na pytanie, pokazując swój tok rozumowania krok po kroku.
Używaj TYLKO informacji zawartych w kontekście.

Kontekst:
{context}

Pytanie:
{question}

Rozumowanie:
1. Najpierw zidentyfikuję kluczowe informacje z kontekstu
2. Następnie przeanalizuję, jak te informacje odnoszą się do pytania
3. Na koniec sformułuję odpowiedź

Krok 1 - Kluczowe informacje:
[Tutaj wymień najważniejsze fakty z kontekstu]

Krok 2 - Analiza:
[Tutaj pokaż, jak te fakty pomagają odpowiedzieć na pytanie]

Krok 3 - Odpowiedź końcowa:
[Tutaj podaj finalną odpowiedź]"""