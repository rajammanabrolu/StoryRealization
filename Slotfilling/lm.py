from nltk.lm import MLE
import pickle
from nltk.corpus import wordnet as wn
import random
import en

class LM:
	"""
	def __init__(self):
		self.lm = pickle.load(open("Models/wordnetLM.pickle",'rb'))
	"""	

	def pickNoun(self, cat, sent):
		if sent: prev = sent[-1]
		#print(prev, cat)
		try:
			_,category,_ = cat.split("'")
			category = category.split(".")[0]
		except:
			return cat
		synset = wn.synsets(category)
#		print(synset)
		hypo = []
		for syn in synset:
#			print("SYN:",syn)
#			print("CAT:",cat)
			if str(syn) == cat:				
				hypo = list(syn.hyponyms())
#				print("HYPO:",hypo)
				break
		
		"""
		skinned_hypo = [str(x).split("'")[1].split(".")[0] for x in hypo]
		split_hypo = [x.split("_")[0] for x in skinned_hypo]
		print(split_hypo)
		print(self.lm.generate(num_words=1000,text_seed=prev))
		for guess in self.lm.generate(num_words=1000,text_seed=prev):
			try:
				guess = en.verb.present(x, person=1)
				if guess in split_hypo:
					i = split_hypo.index(guess)
					return " ".join(skinned_hypo[i].split("_"))
			except:
				continue

		"""		

		if hypo:
			random.shuffle(hypo)
			s = str(hypo.pop())
			selection = s.split("'")[1]
			#_,selection,_ = str(hypo.pop()).split("'")
			return selection.split(".")[0].replace("_"," ")
		else:
			return category.replace("_"," ")


		
