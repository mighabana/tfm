from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and prepare questions from the QASPER dataset")
    parser.add_argument('--model', nargs='?', type=str, required=True, help="Input the embedding model path or huggingface repository name")
    parser.add_argument('--training_data_path', nargs='?', type=str, required=True, help="Input the path to qasper text data")
    parser.add_argument('--output_path', nargs='?', type=str, required=True, help="Input the output file path")
    args = parser.parse_args()

    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer(args.model, trust_remote_code=True)

    # Define your train examples. You need more than just two examples...
    train = pd.read_parquet(args.training_data_path)
    train_examples = []

    for idx, row in train.iterrows():
        example = InputExample(texts=[row['anchor'], row['positive'], row['negative']])
        train_examples.append(example)

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.TripletLoss(model)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    
    model.save_pretrained(args.output_path)