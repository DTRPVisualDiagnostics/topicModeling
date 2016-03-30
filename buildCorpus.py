import logging, os, csv, re, pdb

from collections import defaultdict

from gensim import corpora, models, similarities

from nltk.stem.porter import PorterStemmer

from stop_words import get_stop_words

p_stemmer = PorterStemmer()

# Create English stop words list
en_stop = get_stop_words('en')
en_stop += ["yeah","ok","like","mhm","err","think", "xxx", "you", "eh", "ehm", "huh", "umm", "uhh"]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#(path, transcript column #, delimiter, initial rows to skip)
LIST_OF_TRANSCRIPTS_INFO = [('../../datasets/dschool-dataset/csv', 3, ',', 1), ('../../datasets/DTRS2015-dataset/tsv', 1, '\t', 0), ('../../datasets/DTRS2016-dataset/csv', 3, ',', 6)]

def buildCorpus():
	documents = []

	for info in LIST_OF_TRANSCRIPTS_INFO:		
		for transcriptPath in os.listdir(info[0]):
			document = loadFileIntoList(info[0] + "/" + transcriptPath, info[1], info[2], info[3])
			documents.append(document)

	frequency = defaultdict(int)
	for text in documents:
	    for token in text:
	        frequency[token] += 1

	texts = [[token for token in text if frequency[token] > 1] for text in documents]

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
	doc = " ".join(doc).split(" ")
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
	stemmed_line = " ".join(stemmed_tokens)
	return stemmed_line

buildCorpus()