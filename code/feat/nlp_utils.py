
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVector(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class StemmedCountVector(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def tfidf_vec(ngram_r=(1,1)):
    return StemmedTfidfVector(ngram_range=ngram_r,stop_words='english')

def bow_vec(ngram_r=(1,1)):
    return StemmedCountVector(ngram_range=ngram_r,stop_words='english')