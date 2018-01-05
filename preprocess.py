# Preprocessing Europarl v0.7
import sys

reload(sys)
sys.setdefaultencoding('utf8')

from nltk.tokenize import sent_tokenize
print "reading input files"
f_en_in = open('hello_copy.csv','r+')
text = f_en_in.read()
f_en_in.close()
S_en = sent_tokenize(text.decode('utf8'))
f_fr_in = open('french_copy.csv','r+')
text = f_fr_in.read()
f_fr_in.close()
S_fr = sent_tokenize(text.decode('utf8'))
text = ''


print len(S_en)
print len(S_fr)

f_en_out = open('en_trunc.txt', 'w+')
f_fr_out = open('fr_trunc.txt', 'w+')

for i in xrange(len(S_en)):
    s = S_en[i]
    if s.count(' ') < 15:
        f_en_out.write("<BOS> " + s + " <EOS>\n")
        f_fr_out.write("<BOS> " + s + " <EOS>\n")
    if i % 1000 == 0:
        print i
