from typing import Callable
from nltk import download
from nltk.tokenize import word_tokenize

class TFIDF_Embedder():

    def __init__(self,
            corpus: list[str],
            preprocesser:Callable= None,
            tokenizer:Callable= word_tokenize
            ) -> None:
        
        download('punkt')

        # TODO add tf variants
        # TODO add idf variants
        self.tf = self.__tf
        self.idf = self.__idf
        self.tokenizer = tokenizer
        self.preprocesser = preprocesser

        self.corpus = [
            self.__preprocess(document)
            for document in corpus
        ]

        # Build vocab key
        self.vocab = {
            token : index
            for index, token in enumerate(
                # Flatten the corpus into a single list and remove duplicate tokens
                set([token for document in self.corpus for token in document])
            )
        }
        

    def __preprocess(self, document: str) -> list[str]:
        tokens = self.tokenizer(document)
        if self.preprocesser != None:
            tokens = self.preprocesser(tokens)
        return tokens

    def __tf(self, term: str, document: list[str]) -> float:
        return document.count(term) / len(document)


    def __idf(self, term: str) -> float:
        """
        Find the Inverse Document Frequency of a term
        """
        return len(self.corpus) / sum([doc.count(term) for doc in self.corpus])

    def __call__(self, document: str) -> list[float]:
        tokens = self.__preprocess(document)

        vector = [0.0 for _ in range(len(self.vocab))]

        for term in tokens:
            if term not in self.vocab:
                continue
            vector[self.vocab[term]] = self.__tf(term, tokens) * self.__idf(term)
        return vector

if __name__ == '__main__':
    corpus = [
        'the quick brown fox jumped over the lazy dog',
        'peter piper picked a patch of pickled peppers',
        'the lazy dog slept under the quick brown fox',
        'that\'s crazy, I didn\'t know that was a thing'
    ]

    embedder = TFIDF_Embedder(
            corpus=corpus
            )

    print(embedder('that brown dog is very cute'))
