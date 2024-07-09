import xxhash
import spacy
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset
import pandas as pd

class TextProcessor():
    def __init__(self, spacy_model:str='en_core_web_lg', embedding_model:str='thenlper/gte-large', threshold_type:str='percentile'):
        self.nlp = spacy.load(spacy_model, exclude=["parser"])
        self.nlp.enable_pipe("senter")

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.semantic_splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type=threshold_type)

    def sentence_segmentation(self, text:str):
        sentences = []
        doc = self.nlp(text)
        for sent in doc.sents:
            sentences.append(sent)

        sentences = [str(sent).replace('\n','') for sent in sentences if sent]
        ids = [xxhash.xxh64(sent + str(idx), seed=999).hexdigest() for idx, sent in enumerate(sentences)]

        return sentences, ids

    def paragraph_segmentation(self, text:str):
        paragraphs = text.split('\n')
        paragraphs = [paragraph for paragraph in paragraphs if paragraph]
        ids = [xxhash.xxh64(paragraph + str(idx), seed=999).hexdigest() for idx, paragraph in enumerate(paragraphs)]
        return paragraphs, ids

    def semantic_segmentation(self, text, metadata):

        if isinstance(text, str):
            docs = self.semantic_splitter.create_documents([text], metadatas=[metadata])
            docs = [doc.page_content for doc in docs if doc.page_content]
            ids = [xxhash.xxh64(doc + str(idx), seed=999).hexdigest() for idx, doc in enumerate(docs)]
            return docs, ids
        elif isinstance(text, list):
            docs = self.semantic_splitter.create_documents(text, metadatas=metadata)
            docs = [doc.page_content for doc in docs if doc.page_content]
            ids = [xxhash.xxh64(doc + str(idx), seed=999).hexdigest() for idx, doc in enumerate(docs)]
            return docs, ids
        else:
            print('wrong type')
            return [], []

if __name__ == "__main__":
    import argparse
    
    tp = TextProcessor()
    
    parser = argparse.ArgumentParser(description="Download and prepare full text from the QASPER dataset")
    parser.add_argument('--split', nargs='?', type=str, required=True, help="Input dataset split")
    parser.add_argument('--output_path', nargs='?', type=str, required=True, help="Input the output file path")
    args = parser.parse_args()
    
    dataset = load_dataset('allenai/qasper', split=args.split)

    _titles = []
    _abstracts = []
    _sentence_ids = []
    _sentences = []
    _sentence_sections = []
    _paragraphs = []
    _sections = []
    _full_text = []

    for i in range(len(dataset['full_text'])):
        title = dataset['title'][i]
        abstract = dataset['abstract'][i]
        sections = dataset['full_text'][i]['section_name']
        par_section = [(paragraph, sections[idx]) for idx, section in enumerate(dataset['full_text'][i]['paragraphs']) for paragraph in section]
        paragraphs = [ps[0] for ps in par_section]
        sections = [ps[1] for ps in par_section]
        full_text = "\n".join(paragraphs)
        
        sentence_ids = []
        sentences = []
        sentence_sections = []
        for x in range(len(paragraphs)):
            section = sections[x]
            s, ids = tp.sentence_segmentation(paragraphs[x])
            sentence_ids.extend(ids)
            sentences.extend(s)
            sentence_sections.extend([section for i in s])
                
        _titles.append(title)
        _abstracts.append(abstract)
        _sentence_ids.append(sentence_ids)
        _sentences.append(sentences)
        _sentence_sections.append(sentence_sections)
        _paragraphs.append(paragraphs)
        _sections.append(sections)
        _full_text.append(full_text)
        
    df = pd.DataFrame({
        'titles': _titles,
        'abstracts': _abstracts,
        'sentences': _sentences,
        'sentence_sections': _sentence_sections,
        'paragraphs': _paragraphs,
        'sections': _sections,
        'full_text': _full_text
    })
    
    df.to_parquet(args.output_path, index=False)