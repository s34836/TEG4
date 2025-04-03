# Tasks for today

1. Create simple RAG 
2. Go to RAG Techniques Repo
3. Choose 3 pairs of additional modules that can improve our RAG
4. Choose 3 different Prompt Engineering Techniques that can improve our RAG
5. Implement these in Python and create comparison - we should have 6 outputs per input
6. Ask 3 questions to your data and generate dataframe (3 x 6 rows) with evaluation
7. Evaluation will use at least 3 RAGAs metrics, but also send it to new instance of LLM (discriminator)
8. Create streamlit/gradio app that will be a frontend for such comparison - user query, processing, show results

What to remember:
* Do it as your Portfolio project
* Specify functions parameters types
* Remember about error handling
* Remember about SOLID principles
* Create baseline and extend it using modules and prompts


To get csv:
# From the project root (TEG4 directory)
python -m backend.evaluation.rag_evaluation
To run backend:
python -m backend.app
Tu run frontend:
streamlit run frontend/app.py