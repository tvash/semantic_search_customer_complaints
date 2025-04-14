# Sentiment Analysis Using LLM (from CSV Complaints)

import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load CSV complaint data
df = pd.read_csv("data/customer_complaints.csv")

# Step 2: Create prompt template for sentiment classification
prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a helpful assistant for analyzing customer complaints.
Classify the sentiment of the following complaint as Positive, Neutral, or Negative.

Complaint:
{text}

Sentiment:
"""
)

# Step 3: Create LLM chain
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
chain = LLMChain(llm=llm, prompt=prompt)

# Step 4: Analyze sentiment for each complaint
sentiments = []
print("\n🔍 Analyzing sentiment for complaints...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    complaint_text = row.get("Complaint Text", "")
    if not complaint_text.strip():
        sentiments.append("Unknown")
        continue

    result = chain.run(text=complaint_text)
    sentiments.append(result.strip())

# Step 5: Store and export results
df["Sentiment"] = sentiments
output_path = "data/complaints_with_sentiment.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Sentiment analysis completed. Results saved to: {output_path}")
