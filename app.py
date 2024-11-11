from flask import Flask, request, jsonify
from transformers import pipeline
from google.cloud import storage
import os
import pandas as pd

app = Flask(__name__)

def load_dataset_from_cloud_storage(bucket_name, blob_path):
    """Loads the dataset from Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename("/home/nadine_vanderhaar/awesome-chatgpt-prompts/prompts.csv")  

    try:
        df = pd.read_csv("/home/nadine_vanderhaar/awesome-chatgpt-prompts/prompts.csv")
        context = df["Prompt"].to_string(index=False)  # Combine all prompts into one string
        return context
    except FileNotFoundError:
        return "Dataset file not found. Please check the file path."

@app.route('/', methods=['POST'])
def answer_question():
    """Handles POST requests to answer questions."""
    data = request.get_json()
    question = data.get('question')

    if question:
        # Load the dataset from Cloud Storage
        context = load_dataset_from_cloud_storage(
            bucket_name=os.environ.get("BUCKET_NAME"),  
            blob_path=os.environ.get("DATASET_PATH")
        )

        if context == "Dataset file not found. Please check the file path.":
            return jsonify({"error": context}), 400

        question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')
        result = question_answerer(question=question, context=context)
        response = {
            "answer": result["answer"],
            "score": result["score"],
            "start": result["start"],
            "end": result["end"]
        }
        return jsonify(response)
    else:
        return jsonify({"error": "Missing question parameter"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001) 