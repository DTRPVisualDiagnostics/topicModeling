import logging, os, csv, re, pdb

from gensim import corpora, models, similarities, matutils

from nltk.stem.porter import PorterStemmer

from stop_words import get_stop_words

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

p_stemmer = PorterStemmer()

# Create English stop words list
en_stop = get_stop_words('en')
en_stop += ["yeah","ok","like","mhm","err","think", "xxx", "you", "eh", "ehm", "huh", "umm", "uhh"]

lsi = models.LsiModel.load('./tmp/model.lsi')

lda = models.LdaModel.load('./tmp/model.lda')

dictionary = corpora.Dictionary.load('./tmp/DTRP.dict')

corpus = corpora.MmCorpus('./tmp/DTRP.mm')

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

#(path, transcript column #, delimiter, initial rows to skip)
LIST_OF_TRANSCRIPTS_INFO = [('../../datasets/dschool-dataset/csv', 3, ',', 1), ('../../datasets/DTRS2015-dataset/tsv', 1, '\t', 0), ('../../datasets/DTRS2016-dataset/csv', 3, ',', 6)]

def loadDocuments():
	documents = []

	for info in LIST_OF_TRANSCRIPTS_INFO:		
		for transcriptPath in os.listdir(info[0]):
			document = loadFileIntoList(info[0] + "/" + transcriptPath, info[1], info[2], info[3])
			documents.append(document)
	
	for doc in documents:
		last_sentence_lsi = []
		last_sentence_lda = []
		for sentence in doc:
			sentence_bow = dictionary.doc2bow(sentence.split(" "))
			print(sentence)
			#print(lsi[tfidf[sentence_bow]])
			#print(lda[sentence_bow])
			sim_lsi = matutils.cossim(lsi[tfidf[sentence_bow]], last_sentence_lsi)
			sim_lda = matutils.cossim(lda[sentence_bow], last_sentence_lda)
			print(sim_lsi)
			print(sim_lda)
			last_sentence_lsi = lsi[tfidf[sentence_bow]]
			last_sentence_lda = lda[sentence_bow]

def loadFileIntoList(path, index, delimiter, skipLines):
	doc = []
	with open(path, 'rt') as csvfile:
		csvReader = csv.reader(csvfile, delimiter=delimiter)
		for x in range(skipLines):
			next(csvReader,None)
		for row in csvReader:
			if len(row) >= index and row[index]:
				cleaned_line = cleanLine(row[index])
				if (cleaned_line != ""):
					doc.append(cleaned_line.rstrip('\n'))
	return doc

def cleanLine(line):
	lower_line = line.lower()
	cleaned_line = re.sub(r'\([^)]*\)', ' ', lower_line)
	cleaned_line = re.sub(r'\[[^\]]*\]', ' ', cleaned_line)
	cleaned_line = re.sub(r'\-[^-]*\-', ' ', cleaned_line)
	cleaned_line = cleaned_line.replace(":"," ")
	cleaned_line = cleaned_line.replace("."," ")
	cleaned_line = cleaned_line.replace("!"," ")
	cleaned_line = cleaned_line.replace("?"," ")
	cleaned_line = cleaned_line.replace(","," ")
	cleaned_line = cleaned_line.replace("-"," ")
	cleaned_line = cleaned_line.replace("_"," ")
	cleaned_line = cleaned_line.replace("“"," ")
	cleaned_line = cleaned_line.replace("\""," ")
	cleaned_line = cleaned_line.replace("\'"," ")
	cleaned_line = cleaned_line.replace("<"," ")
	cleaned_line = cleaned_line.replace(">"," ")
	tokens = cleaned_line.split(" ")
	stopped_tokens = [i for i in tokens if not i in en_stop]
	stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens if i]
	final_tokens = [i for i in stemmed_tokens if i != "s" and i != "t"]
	final_line = " ".join(final_tokens)
	return final_line

loadDocuments()