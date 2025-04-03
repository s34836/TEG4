import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import json
import traceback
import plotly.express as px
import plotly.graph_objects as go
import requests
import time

# Load environment variables
dotenv_path = os.path.join('../', 'config', '.env')
load_dotenv(dotenv_path)
api_key = os.getenv('OPENAI_API_KEY_TEG')
os.environ["OPENAI_API_KEY"] = api_key

# API endpoint
API_URL = "http://localhost:5001/api"

# Configure the page
st.set_page_config(
    page_title="RAG System Comparison",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 RAG System Comparison")
st.markdown("""
This app allows you to compare different RAG (Retrieval-Augmented Generation) system variants.
Enter your question and optionally provide ground truth to evaluate the system's performance.

This RAG uses: Hebda, I. (2020). Konsekwencje nagradzania wyników pracy nauczycieli szkół publicznych. PWN.

⚠️ Note: The evaluation process can take several minutes as it:
- Generates answers using different RAG variants
- Evaluates each answer using multiple metrics
- Runs LLM-based evaluations
""")

# Input section
col1, col2 = st.columns(2)

with col1:
    question = st.text_area("Enter your question:", 
                           placeholder="e.g., Co to jest motywacja?",
                           height=100)

with col2:
    ground_truth = st.text_area("Enter ground truth:",
                               placeholder="e.g., Motywacja to wewnętrzny lub zewnętrzny impuls pobudzający człowieka do działania i wytrwałości w dążeniu do określonego celu.",
                               height=100)

# Submit button
if st.button("Evaluate RAG System", type="primary"):
    if not question:
        st.error("Please enter a question")
    else:
        try:
            # Check if server is running
            server_ok = True
            try:
                health_check = requests.get(f"{API_URL.replace('/api', '')}/test")
                if health_check.status_code != 200:
                    st.error("Flask server is not responding correctly. Please make sure it's running on http://localhost:5001")
                    st.error(f"Server response: {health_check.text}")
                    server_ok = False
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Flask server. Please make sure it's running on http://localhost:5001")
                server_ok = False
            
            if server_ok:
                # Prepare the request data
                data = {
                    "question": question,
                    "ground_truth": ground_truth if ground_truth else None
                }
                
                st.info("Sending request to server...")
                st.json(data)  # Show the request data
                
                # Create a status message
                status = st.empty()
                status.markdown("### 🔄 Processing your request...")
                status.markdown("""
                Please wait while we:
                - Generate answers using different RAG variants
                - Evaluate answer quality and relevance
                - Calculate metrics and prepare visualizations
                
                This may take several minutes...
                """)
                
                # Send request to Flask backend
                try:
                    response = requests.post(
                        f"{API_URL}/evaluate",
                        json=data,
                        timeout=300  # 5 minutes timeout
                    )
                    
                    if response.status_code == 200:
                        # Clear status message
                        status.empty()
                        
                        results = response.json()
                        st.success("✅ Evaluation completed successfully!")
                        
                        # Display results in tabs
                        tab1, tab2, tab3, tab4 = st.tabs(["Answers", "Metrics", "Context", "Detailed Analysis"])
                        
                        with tab1:
                            st.subheader("Generated Answers")
                            answers_df = pd.DataFrame(results["results"])
                            st.dataframe(answers_df[["variant", "answer"]], use_container_width=True)
                        
                        with tab2:
                            st.subheader("Evaluation Metrics")
                            
                            # Prepare metrics data
                            metrics_df = pd.DataFrame(results["results"])
                            metrics = ["answer_relevancy", "faithfulness", "answer_correctness", "llm_score"]
                            
                            # Create bar chart for metrics
                            fig = go.Figure()
                            for metric in metrics:
                                fig.add_trace(go.Bar(
                                    name=metric.replace("_", " ").title(),
                                    x=metrics_df["variant"],
                                    y=metrics_df[metric],
                                    text=metrics_df[metric].round(3),
                                    textposition='auto',
                                ))
                            
                            fig.update_layout(
                                title="RAG System Metrics Comparison",
                                xaxis_title="RAG Variant",
                                yaxis_title="Score",
                                barmode='group',
                                yaxis_range=[0, 1],
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display metrics table
                            st.dataframe(metrics_df[["variant"] + metrics].round(3), use_container_width=True)
                            
                            # Create radar chart for each variant
                            st.subheader("Metrics Radar Chart")
                            radar_fig = go.Figure()
                            
                            for _, row in metrics_df.iterrows():
                                radar_fig.add_trace(go.Scatterpolar(
                                    r=[row[m] for m in metrics],
                                    theta=[m.replace("_", " ").title() for m in metrics],
                                    name=row["variant"],
                                    fill='toself'
                                ))
                            
                            radar_fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )
                                ),
                                showlegend=True,
                                height=500
                            )
                            
                            st.plotly_chart(radar_fig, use_container_width=True)
                        
                        with tab3:
                            st.subheader("Retrieved Context")
                            for _, row in metrics_df.iterrows():
                                with st.expander(f"{row['variant']} - Context"):
                                    st.text(row["context"])
                        
                        with tab4:
                            st.subheader("LLM Evaluations")
                            for _, row in metrics_df.iterrows():
                                with st.expander(f"{row['variant']} - Score: {row['llm_score']:.3f}"):
                                    st.write(row["llm_justification"])
                            
                            if ground_truth:
                                st.subheader("Ground Truth")
                                st.info(ground_truth)
                    
                    else:
                        status.empty()
                        st.error(f"Error from server (Status {response.status_code}):")
                        st.error(response.text)
                        st.error("Request data:")
                        st.json(data)
                
                except requests.exceptions.Timeout:
                    status.empty()
                    st.warning("⚠️ The request is still processing...")
                    st.info("The evaluation process can take several minutes. You can wait or try again later.")
                except requests.exceptions.RequestException as e:
                    status.empty()
                    st.error(f"Request failed: {str(e)}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Full traceback:")
            st.error(traceback.format_exc())