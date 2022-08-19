import csv
import nltk
import re
import string
inpTweets = csv.reader(open('dataset_v1.csv', 'rb'),  delimiter=',', quotechar='"', escapechar='\\')
testTweets= csv.reader(open('dataset_v1.csv', 'rb'),  delimiter=',', quotechar='"', escapechar='\\')

def processTweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    tweet= tweet.translate(string.maketrans("",""), string.punctuation)
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return tweet

def biggest(a, y, z):
	l=max(a,y,z)
	if l==a:
		return 1
	elif l==y:
		return 2
	else:
		return 3
	
def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('rt')
    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
sw = getStopWordList('stopwords.txt')
lis={}
c_oth=0
c_ind=0
c_mob=0
oth_words=0
ind_words=0
mob_words=0

c=0
#Training Process
for row in inpTweets:
    c+=1
    if c==1201:
        break
    else:
	cls = int(row[1])
	t = row[0]
	t1 = processTweet(t)					#removing RTs, @, etc
	twt = nltk.tokenize.wordpunct_tokenize(t1)
	for i in twt[:]:							#removing stop words
		if i in sw :
			twt.remove(i)
	#creating dictionary where word is the key and list with length 3 as it's value where list[0] represent class 1(others), list[1] represents class 2(Indian National Congress), list[2] represents classs 3(mobile congress)
	for i in twt:							
		if i not in lis.keys():
			lis[i]=[1,1,1]
		if cls==1:
		        oth_words+=1
			c_oth+=1		#counting number of tweets labelled as 1
			lis[i][0]+=1	#updating lis[word] for class 1
		elif cls==2:
		        ind_words+=1
			c_ind+=1
			lis[i][1]+=1
		elif cls==3:
		        mob_words+=1
			c_mob+=1
			lis[i][2]+=1

#finding likelihood for each word in dictionary for each class
for row in lis.keys():
		t1=lis[row][0]
		t2=lis[row][1]	
		t3=lis[row][2]
		lis[row][0]=float(lis[row][0])/(t1+t2+t3)
		lis[row][1]=float(lis[row][1])/(t1+t2+t3)
		lis[row][2]=float(lis[row][2])/(t1+t2+t3)

#Prior probability for each class
pri_ot=float(c_oth)/(c_oth+c_ind+c_mob)
pri_in=float(c_ind)/(c_oth+c_ind+c_mob)
pri_mo=float(c_mob)/(c_oth+c_ind+c_mob)

print "Prior probability for class 1(Others):",pri_ot
print "Prior probability for class 2(INC):",pri_in
print "Prior probability for class 3(Mobile congress):",pri_mo

#Testing Phase
c=0
c_test=0
c_ini=0
for row in testTweets:
    c_ini+=1
    if c_ini>1200:
	k=1			#represents posterior prob for class1
	l=1			#represents posterior prob for class2
	m=1			#represents posterior prob for class3
	cls = int(row[1])
	t = row[0]
	t1 = processTweet(t)
	twt = nltk.tokenize.wordpunct_tokenize(t1)
	for wrd in twt:			#finding posterior proability for each tweet
		if wrd in lis:
			k*=lis[wrd][0]		
			l*=lis[wrd][1]
			m*=lis[wrd][2]
		else:
		        k*=1.0/oth_words
		        l*=1.0/ind_words
		        m*=1.0/mob_words
		
	mx = biggest(k*pri_ot,l*pri_in,m*pri_mo)
	if(mx==cls):		#checking for accuracy
		c=c+1
	c_test+=1
print "Number of tweets which were predicted accurately:",c
print "Total number of tweets:",c_test
acc=float(c)/c_test
print "Accuracy %:",acc*100

