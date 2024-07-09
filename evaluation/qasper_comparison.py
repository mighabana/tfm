import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

def abstractive_evaluation(evaluation_df, experiment):
    abstractive_eval = evaluation_df[evaluation_df['answers'].str.strip() != '']

    if experiment == 'crossencoder':
        abstractive_eval['evidence_scores'] = abstractive_eval.apply(lambda x: scorer.score("\n".join(x['evidences']), "\n".join(x['crossencoder_evidences']) ), axis = 1)
    else:
        abstractive_eval['evidence_scores'] = abstractive_eval.apply(lambda x: scorer.score("\n".join(x['evidences']), "\n".join(x['retrieval_evidences']) ), axis = 1)
    abstractive_eval['answer_scores'] = abstractive_eval.apply(lambda x: scorer.score(x['answers'], x['llm_answers']), axis=1)

    abstractive_eval['evidence_precision'] = abstractive_eval.apply(lambda x: x['evidence_scores']['rouge1'][0], axis=1)
    abstractive_eval['answer_precision'] = abstractive_eval.apply(lambda x: x['answer_scores']['rouge1'][0], axis=1)
    abstractive_eval['evidence_recall'] = abstractive_eval.apply(lambda x: x['evidence_scores']['rouge1'][1], axis=1)
    abstractive_eval['answer_recall'] = abstractive_eval.apply(lambda x: x['answer_scores']['rouge1'][1], axis=1)
    abstractive_eval['evidence_f1'] = abstractive_eval.apply(lambda x: x['evidence_scores']['rouge1'][2], axis=1)
    abstractive_eval['answer_f1'] = abstractive_eval.apply(lambda x: x['answer_scores']['rouge1'][2], axis=1)

    qs = abstractive_eval['questions'].unique()

    evidence_eval = []
    answer_eval = []
    for q in qs:
        evidence = abstractive_eval.query('questions == @q').sort_values('evidence_f1', ascending=False).iloc[0]
        evidence_eval.append(evidence)
        
        answer = abstractive_eval.query('questions == @q').sort_values('answer_f1', ascending=False).iloc[0]
        answer_eval.append(answer)
        
    abstractive_evidence_eval_df = pd.DataFrame(evidence_eval)
    abstractive_answer_eval_df = pd.DataFrame(answer_eval)
    
    return abstractive_evidence_eval_df, abstractive_answer_eval_df

def extractive_evaluation(evaluation_df, experiment):
    extractive_eval = evaluation_df.query('extractive_span_len > 0')

    if experiment == 'crossencoder':
        extractive_eval['evidence_scores'] = extractive_eval.apply(lambda x: scorer.score("\n".join(x['evidences']), "\n".join(x['crossencoder_evidences'])), axis = 1)
    else:
        extractive_eval['evidence_scores'] = extractive_eval.apply(lambda x: scorer.score("\n".join(x['evidences']), "\n".join(x['retrieval_evidences'])), axis = 1)
    extractive_eval['answer_scores'] = extractive_eval.apply(lambda x: scorer.score("\n".join(x['extractive_spans']), x['llm_answers']), axis=1)


    extractive_eval['evidence_precision'] = extractive_eval.apply(lambda x: x['evidence_scores']['rouge1'][0], axis=1)
    extractive_eval['answer_precision'] = extractive_eval.apply(lambda x: x['answer_scores']['rouge1'][0], axis=1)
    extractive_eval['evidence_recall'] = extractive_eval.apply(lambda x: x['evidence_scores']['rouge1'][1], axis=1)
    extractive_eval['answer_recall'] = extractive_eval.apply(lambda x: x['answer_scores']['rouge1'][1], axis=1)
    extractive_eval['evidence_f1'] = extractive_eval.apply(lambda x: x['evidence_scores']['rouge1'][2], axis=1)
    extractive_eval['answer_f1'] = extractive_eval.apply(lambda x: x['answer_scores']['rouge1'][2], axis=1)
    
    qs = extractive_eval['questions'].unique()

    evidence_eval = []
    answer_eval = []
    for q in qs:
        evidence = extractive_eval.query('questions == @q').sort_values('evidence_f1', ascending=False).iloc[0]
        evidence_eval.append(evidence)
        
        answer = extractive_eval.query('questions == @q').sort_values('answer_f1', ascending=False).iloc[0]
        answer_eval.append(answer)
        
    extractive_evidence_eval_df = pd.DataFrame(evidence_eval)
    extractive_answer_eval_df = pd.DataFrame(answer_eval)
    
    return extractive_evidence_eval_df, extractive_answer_eval_df

def yes_no_evaluation(evaluation_df):
    yes_no_eval = evaluation_df[evaluation_df['yes_no'].notna()]

    yes_no_eval['evidence_scores'] = yes_no_eval.apply(lambda x: scorer.score("\n".join(x['evidences']), "\n".join(x['retrieval_evidences'])), axis = 1)
    yes_no_eval['yes_no_answer'] = yes_no_eval.apply(lambda x: 'yes' if x['yes_no'] else 'no', axis=1)
    yes_no_eval['answer_scores'] = yes_no_eval.apply(lambda x: scorer.score(x['yes_no_answer'], x['llm_answers'].lower()), axis=1)


    yes_no_eval['evidence_precision'] = yes_no_eval.apply(lambda x: x['evidence_scores']['rouge1'][0], axis=1)
    yes_no_eval['answer_precision'] = yes_no_eval.apply(lambda x: x['answer_scores']['rouge1'][0], axis=1)
    yes_no_eval['evidence_recall'] = yes_no_eval.apply(lambda x: x['evidence_scores']['rouge1'][1], axis=1)
    yes_no_eval['answer_recall'] = yes_no_eval.apply(lambda x: x['answer_scores']['rouge1'][1], axis=1)
    yes_no_eval['evidence_f1'] = yes_no_eval.apply(lambda x: x['evidence_scores']['rouge1'][2], axis=1)
    yes_no_eval['answer_f1'] = yes_no_eval.apply(lambda x: x['answer_scores']['rouge1'][2], axis=1)

    qs = yes_no_eval['questions'].unique()

    evidence_eval = []
    answer_eval = []
    for q in qs:
        evidence = yes_no_eval.query('questions == @q').sort_values('evidence_f1', ascending=False).iloc[0]
        evidence_eval.append(evidence)
        
        answer = yes_no_eval.query('questions == @q').sort_values('answer_f1', ascending=False).iloc[0]
        answer_eval.append(answer)
        
    yes_no_evidence_eval_df = pd.DataFrame(evidence_eval)
    yes_no_answer_eval_df = pd.DataFrame(answer_eval)

    return yes_no_evidence_eval_df, yes_no_answer_eval_df

def unanswerable_evaluation(evaluation_df):
    unanswerable_eval = evaluation_df.query('unanswerable == True')

    unanswerable_eval['evidence_scores'] = unanswerable_eval.apply(lambda x: scorer.score("\n".join(x['evidences']), "\n".join(x['retrieval_evidences'])), axis = 1)
    unanswerable_eval['answer_scores'] = unanswerable_eval.apply(lambda x: scorer.score("i cannot answer the question", x['llm_answers'].lower()), axis=1)

    unanswerable_eval['evidence_precision'] = unanswerable_eval.apply(lambda x: x['evidence_scores']['rouge1'][0], axis=1)
    unanswerable_eval['answer_precision'] = unanswerable_eval.apply(lambda x: x['answer_scores']['rouge1'][0], axis=1)
    unanswerable_eval['evidence_recall'] = unanswerable_eval.apply(lambda x: x['evidence_scores']['rouge1'][1], axis=1)
    unanswerable_eval['answer_recall'] = unanswerable_eval.apply(lambda x: x['answer_scores']['rouge1'][1], axis=1)
    unanswerable_eval['evidence_f1'] = unanswerable_eval.apply(lambda x: x['evidence_scores']['rouge1'][2], axis=1)
    unanswerable_eval['answer_f1'] = unanswerable_eval.apply(lambda x: x['answer_scores']['rouge1'][2], axis=1)

    qs = unanswerable_eval['questions'].unique()

    evidence_eval = []
    answer_eval = []
    for q in qs:
        evidence = unanswerable_eval.query('questions == @q').sort_values('evidence_f1', ascending=False).iloc[0]
        evidence_eval.append(evidence)
        
        answer = unanswerable_eval.query('questions == @q').sort_values('answer_f1', ascending=False).iloc[0]
        answer_eval.append(answer)
        
    unanswerable_evidence_eval_df = pd.DataFrame(evidence_eval)
    unanswerable_answer_eval_df = pd.DataFrame(answer_eval)
    
    return unanswerable_evidence_eval_df, unanswerable_answer_eval_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run default RAG experiment")
    parser.add_argument('--split', nargs='?', choices=['train', 'test', 'validation'], type=str, required=True)
    parser.add_argument('--experiment', nargs='?', choices=['default', 'grounded', 'window', 'paragraph', 'semantic', 'crossencoder', 'finetuned', 's2a', 'final'], type=str, required=True)
    parser.add_argument('--ideal', action='store_true')
    args = parser.parse_args()
    
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    questions = pd.read_parquet(f'../data_collection/data/qasper_questions_{args.split}.parquet')
    results = pd.read_parquet(f'../data_collection/data/gte_{args.split}{"_ideal" if args.ideal else ""}_{args.experiment}.parquet')
    if args.ideal:
        results = results.drop('evidences', axis=1)

    evaluation_df =  pd.merge(results, questions, 'left', on=['questions'])
    evaluation_df['extractive_span_len'] = evaluation_df.apply(lambda x: len(x['extractive_spans']), axis=1)
    
    # Abstractive Evaluation
    abstractive_evidence_eval_df, abstractive_answer_eval_df = abstractive_evaluation(evaluation_df, args.experiment)
    
    # Extractive Evaluation
    extractive_evidence_eval_df, extractive_answer_eval_df = extractive_evaluation(evaluation_df, args.experiment)
    
    # Yes/No Evaluation
    yes_no_evidence_eval_df, yes_no_answer_eval_df = yes_no_evaluation(evaluation_df)
    
    # Unanswerable Evaluation
    unanswerable_evidence_eval_df, unanswerable_answer_eval_df = unanswerable_evaluation(evaluation_df)
    
    abstractive_evidence_eval_df.to_parquet(f'./results/abstractive_evidence_eval_{args.split}{"_ideal" if args.ideal else ""}_{args.experiment}.parquet')
    abstractive_answer_eval_df.to_parquet(f'./results/abstractive_answer_eval_{args.split}{"_ideal" if args.ideal else ""}_{args.experiment}.parquet')
    extractive_evidence_eval_df.to_parquet(f'./results/extractive_evidence_eval_{args.split}{"_ideal" if args.ideal else ""}_{args.experiment}.parquet')
    extractive_answer_eval_df.to_parquet(f'./results/extractive_answer_eval_{args.split}{"_ideal" if args.ideal else ""}_{args.experiment}.parquet')
    yes_no_evidence_eval_df.to_parquet(f'./results/yes_no_evidence_eval_{args.split}{"_ideal" if args.ideal else ""}_{args.experiment}.parquet')
    yes_no_answer_eval_df.to_parquet(f'./results/yes_no_answer_eval_{args.split}{"_ideal" if args.ideal else ""}_{args.experiment}.parquet')
    unanswerable_evidence_eval_df.to_parquet(f'./results/unanswerable_evidence_eval_{args.split}{"_ideal" if args.ideal else ""}_{args.experiment}.parquet')
    unanswerable_answer_eval_df.to_parquet(f'./results/unanswerable_answer_eval_{args.split}{"_ideal" if args.ideal else ""}_{args.experiment}.parquet')
    
    overall_evidence_f1 = []
    overall_answer_f1 = []
    overall_evidence_recall = []
    overall_answer_recall = []
    
    overall_evidence_f1.extend(abstractive_evidence_eval_df['evidence_f1'].tolist())
    overall_evidence_f1.extend(extractive_evidence_eval_df['evidence_f1'].tolist())
    overall_evidence_f1.extend(yes_no_evidence_eval_df['evidence_f1'].tolist())
    overall_evidence_f1.extend(unanswerable_evidence_eval_df['evidence_f1'].tolist())
    
    overall_answer_f1.extend(abstractive_answer_eval_df['answer_f1'].tolist())
    overall_answer_f1.extend(extractive_answer_eval_df['answer_f1'].tolist())
    overall_answer_f1.extend(yes_no_answer_eval_df['answer_f1'].tolist())
    overall_answer_f1.extend(unanswerable_answer_eval_df['answer_f1'].tolist())
    
    overall_answer_recall.extend(abstractive_answer_eval_df['answer_recall'].tolist())
    overall_answer_recall.extend(extractive_answer_eval_df['answer_recall'].tolist())
    overall_answer_recall.extend(yes_no_answer_eval_df['answer_recall'].tolist())
    overall_answer_recall.extend(unanswerable_answer_eval_df['answer_recall'].tolist())
    
    overall_evidence_recall.extend(abstractive_evidence_eval_df['evidence_recall'].tolist())
    overall_evidence_recall.extend(extractive_evidence_eval_df['evidence_recall'].tolist())
    overall_evidence_recall.extend(yes_no_evidence_eval_df['evidence_recall'].tolist())
    overall_evidence_recall.extend(unanswerable_evidence_eval_df['evidence_recall'].tolist())
    
    
    print(f"""\t\t\t\tAbstractive\t\tExtractive\t\tYes/No\t\tUnanswerable\t\tOverall
        Evidence F1:\t{abstractive_evidence_eval_df['evidence_f1'].mean()}\t\t{extractive_evidence_eval_df['evidence_f1'].mean()}\t\t{yes_no_evidence_eval_df['evidence_f1'].mean()}\t\t{unanswerable_evidence_eval_df['evidence_f1'].mean()}\t\t{np.mean(overall_evidence_f1)}
        Evidence Recall:\t{abstractive_evidence_eval_df['evidence_recall'].mean()}\t\t{extractive_evidence_eval_df['evidence_recall'].mean()}\t\t{yes_no_evidence_eval_df['evidence_recall'].mean()}\t\t{unanswerable_evidence_eval_df['evidence_recall'].mean()}\t\t{np.mean(overall_evidence_recall)}
        Answer F1:\t{abstractive_answer_eval_df['answer_f1'].mean()}\t\t{extractive_answer_eval_df['answer_f1'].mean()}\t\t{yes_no_answer_eval_df['answer_f1'].mean()}\t\t{unanswerable_answer_eval_df['answer_f1'].mean()}\t\t{np.mean(overall_answer_f1)}
        Answer Recall:\t{abstractive_answer_eval_df['answer_recall'].mean()}\t\t{extractive_answer_eval_df['answer_recall'].mean()}\t\t{yes_no_answer_eval_df['answer_recall'].mean()}\t\t{unanswerable_answer_eval_df['answer_recall'].mean()}\t\t{np.mean(overall_answer_recall)}
        """)