import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from transformers import pipeline
import csv

df = pd.read_csv("lender_list.csv", usecols=["Lender"])
df.dropna(subset=['Lender'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv("lender.csv",header=True, columns=df, index=False)

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("mdarhri00/named-entity-recognition")
model = AutoModelForTokenClassification.from_pretrained("mdarhri00/named-entity-recognition")

# Create a pipeline for named entity recognition
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Sample data
data = df.values.tolist()

# Function to classify names
def classify_name(name):
    # Get predictions
    ner_results = ner_pipeline(name)
    # Check if there are any entities of type "PER" (person)
    for entity in ner_results:
        if entity['entity'] == 'B-ORG' or entity['entity'] == 'I-ORG':
            return 'Company Name'
    return 'Human Name'

# Classify sample data
with open('lender_comapny.csv', 'w') as f:
    for name in data:
        name = name[0]
        classification = classify_name(name)
        print(f"{name}: {classification}")

        if classification == "Company Name":
            f.write(name)
            f.write("\n")