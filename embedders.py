import fasttext.util
import numpy


class FastText():

    def __init__(self, dimension=300):
        if dimension > 300 or dimension < 1:
            raise ValueError("FastText expects a dimension between 1 and 300")
        
        # Download and intialize the pretrained embedding model
        fasttext.util.download_model('en', if_exists='ignore')
        self.model = fasttext.load_model('cc.en.300.bin')
        
        # Reduce dimensionality of the model
        if dimension != 300:
            fasttext.util.reduce_model(self.model, dimension) 
    
    def __call__(self, document: str):
        tokens = fasttext.tokenize(document)
        vectors = [
                self.model.get_word_vector(token)
                for token in tokens
                ]
        # Create document prototype embedding by average the word embeddings row-wise
        document_prototype = numpy.column_stack(vectors).mean(axis=1)
        return document_prototype
