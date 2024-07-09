import pandas as pd
import spacy

nlp = spacy.load('en_core_web_lg', exclude=["parser"])
nlp.enable_pipe("senter")

def sentence_segmentation(text:str):
        sentences = []
        doc = nlp(text)
        for sent in doc.sents:
            sentences.append(sent)

        sentences = [str(sent).replace('\n','') for sent in sentences if sent]

        return sentences

def sentence_segmentation_list(text_list:list):
    sentences = []
    for text in text_list:
        doc = nlp(text)
        for sent in doc.sents:
            sentences.append(sent)
            
    sentences = [str(sent).replace('\n','') for sent in sentences if sent]
    
    return sentences
    
def combine_lists(group):
    """Combines lists within a group, handling non-hashable elements."""
    combined_list = []
    for item in group:
        combined_list.extend(item)  # Extend with elements from each list
        
    deduplicated_list = list(set(combined_list))
    return deduplicated_list

def source_mapping(retrieval_evidences, sentence_mapping):
    retrieval_sources = []
    for evidence in retrieval_evidences:
        title = sentence_mapping.get(evidence)
        retrieval_sources.append(title)
        
    return retrieval_sources

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run default RAG experiment")
    parser.add_argument('--split', nargs='?', choices=['train', 'test', 'validation'], type=str, required=True)
    parser.add_argument('--experiment', nargs='?', choices=['default', 'grounded', 'window', 'paragraph', 'semantic', 'crossencoder', 'finetuned', 'final', 's2a'], type=str, required=True)
    args = parser.parse_args()

    text_df = pd.read_parquet(f'../data_collection/data/qasper_text_{args.split}.parquet')
    questions_df = pd.read_parquet(f'../data_collection/data/qasper_questions_{args.split}.parquet')
    
    a = pd.read_parquet(f'./results/abstractive_answer_eval_{args.split}_{args.experiment}.parquet')
    b = pd.read_parquet(f'./results/extractive_answer_eval_{args.split}_{args.experiment}.parquet')

    evaluation_df = pd.concat([a,b])

    titles = []
    sentences = []

    for idx, row in text_df.iterrows():
        sents = row['sentences']
        ts = [row['titles']] * len(sents)
        
        titles.extend(ts)
        sentences.extend(sents)
        
    sentence_mapping = dict(zip(sentences, titles))
    
    evaluation = questions_df.groupby(['titles','questions'])['evidences'].apply(combine_lists)
    
    keys = evaluation.keys()

    updated_evaluation = []

    for idx, evidences in enumerate(evaluation):
        sent_evidences = []
        for evidence in evidences:
            sents = sentence_segmentation(evidence)
            sent_evidences.extend(sents)
        row = {'title': keys[idx][0], 'question': keys[idx][1], 'evidences': sent_evidences}
        updated_evaluation.append(row)
        
    test_df = pd.DataFrame(updated_evaluation)
    
    evaluation_df = pd.merge(evaluation_df, test_df, 'left', left_on=['titles_x', 'questions'], right_on=['title', 'question'])
    
    if args.experiment in ['window', 'paragraph', 'semantic']:
        evaluation_df['retrieval_evidences'] = evaluation_df['retrieval_evidences'].apply(sentence_segmentation_list)
        
    if args.experiment in ['crossencoder']:
        evaluation_df['retrieval_sources'] = evaluation_df['crossencoder_evidences'].apply(lambda x: source_mapping(x, sentence_mapping))
        evaluation_df['retrieval_accuracy'] = evaluation_df.apply(lambda x: [1 if evidence in x['evidences_y'] else 0 for evidence in x['crossencoder_evidences'] ]  ,axis=1)
    
    else:
        evaluation_df['retrieval_sources'] = evaluation_df['retrieval_evidences'].apply(lambda x: source_mapping(x, sentence_mapping))
        evaluation_df['retrieval_accuracy'] = evaluation_df.apply(lambda x: [1 if evidence in x['evidences_y'] else 0 for evidence in x['retrieval_evidences'] ]  ,axis=1)
    
    evaluation_df['retrieval_source_accuracy'] = evaluation_df.apply(lambda x : [1 if x['titles_y'] == source else 0 for source in x['retrieval_sources']] , axis=1)

    evaluation_df.to_parquet(f'./results/evidence_evaluation_{args.split}_{args.experiment}.parquet')
    
    acc = [test for tests in evaluation_df['retrieval_accuracy'].tolist() for test in tests]
    
    acc_s = [test for tests in evaluation_df['retrieval_source_accuracy'].tolist() for test in tests]
    
    print(f"Accuracy: {sum(acc)} / {len(acc)} = {sum(acc)/len(acc)}")
    print(f"Source Accuracy: {sum(acc_s)} / {len(acc_s)} = {sum(acc_s)/len(acc_s)}")