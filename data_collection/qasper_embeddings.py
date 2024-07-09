import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and prepare questions from the QASPER dataset")
    parser.add_argument('--model', nargs='?', type=str, required=True, help="Input the embedding model path or huggingface repository name")
    parser.add_argument('--text_data_path', nargs='?', type=str, required=True, help="Input the path to qasper text data")
    parser.add_argument('--output_path', nargs='?', type=str, required=True, help="Input the output file path")
    args = parser.parse_args()
    
    # questions_df = pd.read_parquet('./data_collection/data/qasper_questions_train.parquet')
    text_df = pd.read_parquet(args.text_data_path)
    
    sentences = [sentence for full_text in text_df['sentences'].to_list() for sentence in full_text if sentence.strip() != '']
    # questions = list(questions_df['questions'].unique())
    
    embedder = SentenceTransformer(args.model, trust_remote_code=True)
    document_embeddings = []
    batch_size = 100
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        batch_embeddings = embedder.encode(batch)
        document_embeddings.append(batch_embeddings)
    
    document_embeddings = np.vstack(document_embeddings)

    with open(args.output_path, 'wb') as f:
        pickle.dump(document_embeddings, f)
    
