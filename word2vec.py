import numpy as np
import csv
import operator
import heapq
import random
import Queue


dim = 4
internal = {}	# stores wt vector corresponding to an internal node described by its path
external = {}	# stores path of external node
alpha = 0.1 	# learning rate

def sigmoid(z):
	return 1/(1+np.exp(-z))

class HuffmanTree:

	def __init__(self, l=None, r=None, n=None, p=None):
		self.path = ''
		self.parent = p
		self.right = r
		self.left = l
		if n is not None:
			self.score = n[1]
			self.no = n[0]
		else:
			self.score = l.score + r.score
			self.no = 'nan'

	def __gt__(self,other):
		return self.score > other.score
	def __lt__(self,other):
		return self.score < other.score
	def __ge__(self,other):
		return self.score >= other.score
	def __le__(self,other):
		return self.score <= other.score
	def __str__(self):
		if self.right is None:
			return str(self.no) + ":" + str(self.score) + ":" + str(self.path)
		return ""
		

	def encode(self):
		l = Queue.LifoQueue()
		l.put(self)
		global internal
		global external
		while not l.empty():
			s = l.get()
			if s.no is 'nan':
				internal[s.path] = np.random.uniform(low = -2/(vocab_length+dim), high = 2/(dim+vocab_length), size=(dim,1))
				s.left.path = s.path + '0'
				if s.left.right is None:
					external[s.left.no] = s.left.path
				s.right.path = s.path + '1'
				if s.right.right is None:
					external[s.right.no] = s.right.path
				l.put(s.left)
				l.put(s.right)
f = open("fr_trunc.txt","r")

words = f.read().split()
f.close()
vocab = list(set(words))
vocab_length = len(vocab)
print vocab_length
word_to_index = {} # stores the indices against words.
unigram = {} # stores freq of each word for huffman table
index_to_word = {}
context_count = {} # Used to calculate prob
instance_count = {} # Used to calc prob
weights = [] # WordEmbeddings basically.
#print "line 73"
for i in xrange(vocab_length):
	word_to_index[vocab[i]] = i
	index_to_word[i] = vocab[i]
	weights.append(np.random.uniform(low = -2/(vocab_length+dim), high = 2/(dim+vocab_length), size=(1,dim)))

for w in words:
	unigram[word_to_index[w]] = unigram[word_to_index[w]]+1 if word_to_index[w] in unigram else 1



freq = unigram.items()
#print freq
htList = []
for item in freq:
    htList.append(HuffmanTree(n=item))
heapq.heapify(htList)
i = 0
while len(htList)>1:
	ht1 = heapq.heappop(htList)
	ht2 = heapq.heappop(htList)
	heapq.heappush(htList,HuffmanTree(l=ht1,r=ht2))

h = heapq.heappop(htList);
h.encode()
#print internal
#print external
print "line 109"


def feedforwardNetwork(trigram):
	h = np.mean(w[trigram[0]],w[trigram[2]])
	l = []
	s = external[trigram[1]]
	b = []
	result = 1
	for a in xrange(len(s),0,-1):
		l.append(s[:-a])
		b.append(s[-a])
	for i,j in zip(l,b):
		if j=='0':
			result += np.log(sigmoid(np.dot(internal[i].transpose(),h)))
		else:
			result += np.log(1-sigmoid(np.dot(internal[i].transpose(),h)))
	return result

def backPropNetwork():
	total = len(words)-1
	for i in xrange(1,total):
		if words[i] in ['<BOS>','<EOS>']:
			continue
		t = (word_to_index[words[i-1]], word_to_index[words[i]], word_to_index[words[i+1]])
		h = np.add(weights[t[0]].transpose(),weights[t[2]].transpose())/2
		l = []
		b = []
		delta_w = {}
		s = external[t[1]]
		for a in xrange(len(s)-1,0,-1):
			l.append(s[:-a])
			b.append(1 if s[a]=='0' else 0)
		
		if i%100000 == 0:
			print "{0} percent of current iteration".format(i*100.0/total)
		delta_w = 0
		for i,j in zip(l,b):
			sig = (sigmoid(np.dot(internal[i].transpose(),h))-j)
			delta_w += sig*internal[i]
			internal[i] -= 0.5*alpha*sig*h
		weights[t[0]] -= 0.5*alpha*delta_w.transpose()
		weights[t[2]] -= 0.5*alpha*delta_w.transpose()


for i in xrange(100):
	print "Iteration {0}, {1} words".format(i, len(words)-1)
	backPropNetwork()
	f = open("output1.txt", 'w')
	for i, w in enumerate(weights):
		w = w/np.sqrt(np.dot(w,w.transpose()))
		f.write("{1} : {0} \n".format(w, index_to_word[i]))
	f.close()
	print "Iteration {0}".format(i)
