from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os 

'''
Function: Initialize the Mistral AI LLM
 - Loads the "mistral-small-latest" model
 - Uses API key from environment variable MISTRAL_API_KEY
 - Sets temperature for controlled creativity
'''
def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.3)


'''
Function: Split transcript into manageable chunks
 - Uses RecursiveCharacterTextSplitter
 - Default chunk size: 3000 characters
 - Overlap: 200 characters (to preserve context between chunks)
'''
def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)


'''
Function: Summarize a full transcript
 - Step 1: Summarize each chunk individually ("map step")
 - Step 2: Combine partial summaries into one professional summary ("reduce step")
 - Output: Final bullet-point meeting summary
'''
def summarize(transcript : str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this portion of meeting/seminar/discussion/scenario transcript concisely."),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        (
            "system",
            "You are an expert meeting/seminar/discussion/scenario summarizer. Combine these partial summaries "
            "into one final professional summary in bullet points.",
        ),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)


'''Function: Generate a short professional title
 - Uses transcript text (first 2000 characters for efficiency)
 - Produces a concise title (≤8 words)
 - Returns only the title, no extra text'''
def generate_title(transcipt : str) -> str:
    llm = get_llm()
    
    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the  transcript, generate a short professional title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return title_chain.invoke(transcipt[:2000])