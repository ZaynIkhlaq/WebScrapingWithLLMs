from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import openai

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="YOUR_API_KEY_HERE"
)
template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}"
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the given description: {parse_description}."
    "2. **No Extra Content:** Do not include any extra information that is not relevant to the task."
    "3. **Empty Response:** If you cannot find any relevant information, return an empty string ('')."
    "4. **Direct Data Only:** Your output should only contain the data that is explicitly requested."
)

def parse_with_gpt(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    
    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        response = chain.invoke(
            {"dom_content": chunk, "parse_description": parse_description}
        )
        print(f"Parsed batch {i} of {len(dom_chunks)}")
        
        # Assuming response is an AIMessage object, extract the content
        # Check the structure of the response to extract the text correctly
        if hasattr(response, 'content'):
            parsed_results.append(response.content)  # Extract the content if it's an AIMessage
        else:
            # If response is a list of messages, extract the content from the first message
            parsed_results.append(response[0].content if response else "")  # Adjust based on actual structure

    return "\n".join(parsed_results)  # This should now work without error
