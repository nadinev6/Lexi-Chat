import streamlit as st
from transformers import pipeline

# Load the question-answering model.
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

if __name__ == "__main__":
    st.title("Lexi Chat")

    user_question = st.text_input("Ask me anything:")

    if user_question:
        with st.spinner("Thinking..."):
            try:
                result = question_answerer(question=user_question)  # No context provided
                st.write(f"Answer: {result['answer']}")
                st.write(f"Score: {result['score']:.4f}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
