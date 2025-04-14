# Topic Modeling Using LLM (from CSV Complaints)

import os
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load complaint data from CSV file
df = pd.read_csv("data/customer_complaints.csv")

# Read the predefined list of topics from a text file
# Topics should be comma-separated in a single line
with open("data/topics.txt", "r") as file:
    predefined_topics = [topic.strip() for topic in file.read().strip().split(",")]

# Join topic list into a single string for prompt injection
topic_string = ", ".join(predefined_topics)

# Construct the prompt template that guides the LLM to choose topics
prompt = PromptTemplate(
    input_variables=["text"],
    template=f"""
You are a helpful assistant that classifies customer complaints into topics.
Given the complaint below, choose the most relevant topics from this list:
{topic_string}

Complaint:
{{text}}

Topics (1-3):
"""
)

# Initialize the OpenAI LLM chain using the defined prompt
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
chain = LLMChain(llm=llm, prompt=prompt)

# Process each complaint and classify it using the LLM
print("\n🔍 Extracting topics from complaints using predefined list...")
topics = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    complaint_text = row.get("Complaint Text", "")
    if not complaint_text.strip():
        topics.append("Unknown")
        continue

    result = chain.run(text=complaint_text)
    topics.append(result.strip())

# Append the extracted topics to the DataFrame and save it
output_path = "data/complaints_with_topics.csv"
df["Topics"] = topics
df.to_csv(output_path, index=False)
print(f"\n✅ Topic modeling completed. Results saved to: {output_path}")
