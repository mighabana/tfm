import re
import time
import pandas as pd
import numpy as np
import torch

from langchain import PromptTemplate
from langchain import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFaceHub
from langchain_core.runnables import RunnablePassthrough
from rouge_score import rouge_scorer

def format_docs(docs):
    return "\n\n".join(docs)

def extract_document(input):
    return input[0]

def final_answer_output_parser(response):
    answer = response.split('[/INST]\n')[1]
    return answer

def default_retriever(query):
    query_embeddings = embeddings.encode([query])
    values, indices = torch.topk(embeddings.similarity(query_embeddings, document_embeddings), k)
    context = [semantic_division for i in indices[0] for semantic_division in semantic_divisions if sentences[i] in semantic_division]
    
    return context, values[0].numpy()

def grounded_retriever(query):
    title_pattern = r"<title>(.*?)</title>"
    title = re.search(title_pattern, query).group(1)
    title_subset_df = text_df.query('titles == @title').iloc[0]
    documents = title_subset_df['sentences'].tolist()
    
    query_embeddings = embeddings.encode([query])
    document_embeddings = embeddings.encode(documents)
    
    similarity = embeddings.similarity(query_embeddings, document_embeddings)
    values, indices = torch.topk(similarity, min(k, len(documents)))
    
    context = [semantic_division for i in indices[0] for semantic_division in semantic_divisions if sentences[i] in semantic_division]
    
    return context, values[0].numpy()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run default RAG experiment")
    parser.add_argument('--llm_model', nargs='?', type=str, required=True)
    parser.add_argument('--hf_token',  nargs='?', type=str, required=True)
    parser.add_argument('--embedding_model', nargs='?', type=str, required=True)
    parser.add_argument('--document_embedding_path', nargs='?', type=str, required=True)
    parser.add_argument('--questions_data_path', nargs='?', type=str, required=True)
    parser.add_argument('--text_data_path', nargs='?', type=str, required=True)
    parser.add_argument('--output_path', nargs='?', type=str, required=True)
    args = parser.parse_args()
    

    llm = HuggingFaceHub(
        repo_id=args.llm_model,
        huggingfacehub_api_token=args.hf_token,
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
        },
    )
    
    k = 6
    embeddings = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    document_embeddings = np.load(args.document_embedding_path, allow_pickle=True)
    text_df = pd.read_parquet(args.text_data_path)
    sentences = [sentence for full_text in text_df['sentences'].to_list() for sentence in full_text if sentence.strip() != '']
    semantic_divisions = [semantic_division for semantic_divisions in text_df['semantic_divisions'].to_list() for semantic_division in semantic_divisions ]
    
    
    template = """
    [INST] 
    <>
    Act as an Researcher specializing in the field of Natural Language Processing. Use the following information to answer the question at the end. If you don't know the answer, just say that you don't know.
    <>
    
    Context:
    {context}
    
    Question:
    {question} 
    [/INST]
    """
    
    system2attention_template = """
    [INST]
    <>
        Act as a Researcher specializing in the field of Natural Language Processing. You will be given a prompt below your task is to re-write the prompt and remove any unnecessary information in the prompt. You are not to include any new information and only use the information provided in the original prompt.
    <>
    
    Prompt:
    {prompt}
    
    [/INST]
    """
    
    final_template = """
    [INST]
    {prompt}
    [/INST]
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    s2a_prompt = PromptTemplate(template=system2attention_template, input_variables=["prompt"])
    final_prompt = PromptTemplate(template=final_template, input_variables=["prompt"])

    retrieval_chain = (
        RunnablePassthrough() 
        | grounded_retriever
    )
    
    s2a_chain = (
        {'prompt': {'context': retrieval_chain | extract_document | format_docs, 'question': RunnablePassthrough()} | prompt }
        | s2a_prompt
        | llm
        | final_answer_output_parser 
    )
    
    final_chain = (
        final_prompt
        | llm
        | final_answer_output_parser
    )
    
    df = pd.read_parquet(args.questions_data_path)
    df = df[['titles', 'questions']].drop_duplicates('questions')
    df = df.sample(100, random_state=999)
    
    # answer_scores = []
    llm_answers = []
    # evidence_scores = []
    s2a_prompts = []
    retrieval_evidences = []
    retrieval_scores = []
    # evidence_source = []
    
    for idx, row in df.iterrows():
    
        query = f"For the paper titled <title>{row['titles']}</title>, {row['questions']}"

        evidence_content, retrieval_score = retrieval_chain.invoke(query)
        retrieval_scores.append(retrieval_score)
        
        # evidence_title = [e.metadata['title'] for e in evidence]
        retrieval_evidences.append(evidence_content)
        # evidence_source.append(evidence_title)
        
        try:
            output_s2a_prompt = s2a_chain.invoke(query)
        except:
            print('Waiting...', flush=True)
            time.sleep(3600)
            output_s2a_prompt = s2a_chain.invoke(query)
            
        s2a_prompts.append(output_s2a_prompt)
            
        try:
            answer = final_chain.invoke({'prompt': output_s2a_prompt})
        except:
            print('Waiting...', flush=True)
            time.sleep(3600)
            answer = final_chain.invoke({'prompt': output_s2a_prompt})
            
            
        llm_answers.append(answer)
        
        # e_score = scorer.score(row['evidences'], evidence_str)
        # evidence_scores.append(e_score)
        
        # a_score = scorer.score(row['answers'], answer)
        # answer_scores.append(a_score)
        
    output_df = df.copy(deep=True)
    output_df['llm_answers'] = llm_answers
    # output_df['answer_scores'] = answer_scores
    output_df['s2a_prompts'] = output_s2a_prompt 
    output_df['retrieval_evidences'] = retrieval_evidences
    output_df['retrieval_scores'] = retrieval_scores
    # output_df['evidence_source'] = evidence_source
    # output_df['evidence_scores'] = evidence_scores
    
    output_df.to_parquet(args.output_path)
