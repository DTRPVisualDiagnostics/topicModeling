import gensim, logging
from gensim import corpora, models, similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

dictionary = corpora.Dictionary.load('./tmp/DTRP.dict')
corpus = corpora.MmCorpus('./tmp/DTRP.mm')

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10) # initialize an LSI transformation

lsi.save('./tmp/model.lsi')

lda = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=10, update_every=0, passes=20)

lda.save('./tmp/model.lda')
