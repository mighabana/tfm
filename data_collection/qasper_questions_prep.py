from datasets import load_dataset
import pandas as pd

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and prepare questions from the QASPER dataset")
    parser.add_argument('--split', nargs='?', type=str, required=True, help="Input dataset split")
    parser.add_argument('--output_path', nargs='?', type=str, required=True, help="Input the output file path")
    args = parser.parse_args()
    
    dataset = load_dataset('allenai/qasper', split=args.split)

    rows = []

    for i in range(len(dataset['qas'])):
        title = dataset['title'][i]
        question_data = dataset['qas'][i]
        
        questions = question_data['question']
        question_writer = question_data['question_writer']
        nlp_background = question_data['nlp_background']
        topic_background = question_data['topic_background']
        paper_read = question_data['paper_read']

        for j in range(len(question_data['answers'])):
            answer_data = question_data['answers'][j]['answer']
            for k in range(len(answer_data)):
                answers = answer_data[k]
                
                unanswerable = answers['unanswerable']
                yes_no = answers['yes_no']
                answer = answers['free_form_answer']
                extractive_spans = answers['extractive_spans']
                evidence = answers['evidence']
                
                row = {
                    'titles': title,
                    'questions': questions[j],
                    'question_writers': question_writer[j],
                    'nlp_backgrounds': nlp_background[j],
                    'topic_backgrounds': topic_background[j],
                    'paper_read': paper_read[j],
                    'unanswerable': unanswerable,
                    'yes_no': yes_no,
                    'answers': answer,
                    'extractive_spans': extractive_spans,
                    'evidences': evidence
                }
                
                rows.append(row)
                
    df = pd.DataFrame(rows)
    df.to_parquet(args.output_path)