# -*- coding: utf-8 -*-
#python3
import pickle, datetime, os
import networkx as nx
from numpy import random
from nltk.corpus import wordnet as wn
from collections import defaultdict
from bidict import bidict

def allXinY(word1, word2):
	list1 = word1.split(" ")
	list2 = word2.split(" ")
	for word in list1:
		if word not in list2:
			return False
	return True

def isVerbNet(thing):
	if "-" in thing:
		pos = thing.index("-")
		if thing[pos+1].isdigit():
			return True
	return False


class MemoryGraph:
	def __init__(self, genderlist):
		""" Creating a new event node that keeps track of the previous event and pointers to all things mentioned in this timestep
		"""
		self.gender_list = genderlist #gender_list[name] = set(genders)
		self.graph = nx.DiGraph()
		self.graph.add_node("-")
		self.prev = -1
		self.NE_gender = defaultdict(str)
		self.NEnums = bidict()

		NER_file = [x.split(";") for x in open("data/sci-fi_ner-lowercase-noPeriod-namechange.txt", 'r').readlines()]
		NER_dict = {k.strip(): ("<MISC>" if v.strip()=="O" else "<"+v.strip()+">") for k,v in NER_file} #entity: tag
		self.rev_NER_dict = defaultdict(set) #tag: set(entities)
		for key, value in NER_dict.items():
			self.rev_NER_dict[value].add(key.title())
		del(NER_file)

	def getGender(self,n):
		genders = [x[0] for x in self.gender_list[n]]
		probs = [x[1] for x in self.gender_list[n]]
		try:
			return random.choice(genders,1,p=probs)[0]
		except ValueError:
			return "X"


	def findFreeNameNumber(self, category="<PERSON>"):
		i = 0
		while True:
			if category+str(i) in self.NEnums:
				i+=1
			else:
				return i
		return False

	def pickNE(self,category):
		nameList = self.NE_gender.keys()
		while True:
			name_select = random.choice(list(self.rev_NER_dict[category]))
			if name_select not in nameList:
				gender = "X"
				if category == "<PERSON>":
					for n in name_select.split(" "):
						#print("pickNE")
						if n in self.gender_list:
							gender = self.getGender(n)
							break
				self.NE_gender[name_select] = gender
				return name_select, gender


	def getTagFromName(self, name, category):
		if name in self.NEnums.inverse.keys():
			return self.NEnums.inverse[name]

		for collectedNE in list(self.NE_gender.keys()):
		#for each named entity that we already have under this tag
			if allXinY(name,collectedNE):
				if name in self.NEnums.inverse:
					return self.NEnums.inverse[name]
			elif allXinY(collectedNE,name): #new name is bigger, replace it
				if name in self.NEnums.inverse:
					tag = self.NEnums.inverse[collectedNE]
					self.NEnums.forceput(tag, name)
					self.NE_gender[name] = self.NE_gender[collectedNE]
					return tag

		#we got a new named entity!
		num = self.findFreeNameNumber(category)
		gender = "X"
		if category=="<PERSON>" and name.split(" ")[0] in self.gender_list:
			gender = self.getGender(name.split(" ")[0])
		tag = category+str(num)
		self.NE_gender[name] = gender
		self.NEnums[tag] = name
		self.NEnums.inverse[name] = tag
		return tag


	def getNameFromTag(self, tag):
		if tag in self.NEnums: return self.NEnums[tag]
		else:
			category = tag.split(">")[0]+">"
			name, gender = self.pickNE(category)
			return name


	def checkIfName(self, word, orig_word=None):
		if "<" in word: return None
		if "Synset" in word:
			pos_name = word.split(".")[0].split("'")[1]
			pos_name = pos_name[0].upper()+pos_name[1:]
			if orig_word and pos_name.lower() == orig_word.lower():
				if pos_name in self.gender_list:
					gender = "X"
					if self.gender_list[pos_name][0][1] > .6 or self.gender_list[pos_name][0][1] < .4:
						gender= self.getGender(pos_name)
					newNum = self.findFreeNameNumber()
					category = "<PERSON>"+str(newNum)
					self.NEnums[category] = pos_name
					self.NEnums.inverse[pos_name] = category
					self.NE_gender[pos_name] = gender
					return pos_name
		return None


	def add_event(self, orig_event, gen_event, event_nouns):
		verb = gen_event[1]
		self.prev+=1
		#create a new event node
		eventString = "E"+str(self.prev)
		newEvent = self.graph.add_node(eventString, verb=verb)
		#connect all recently-mentioned things to this node
		for i, thing in enumerate(orig_event):
			if i == 1: continue
			if i == len(orig_event)-1: continue
			if thing == "EmptyParameter": continue
			what = gen_event[i]
			if isVerbNet(what):
				weight = 0.0
			else:
				name = self.checkIfName(what, thing)
				if name:
					thing = name
					what = self.NEnums.inverse[thing]
				if not thing in self.graph.nodes():
					self.graph.add_node(thing, att=event_nouns[thing])
				weight = 0.4
				if "<" in what and what != "<PRP>":
					weight = 1.0
					self.NEnums[what] = thing
					if "<PERSON>" in what and thing not in self.NE_gender:
						gender= self.getGender(thing)
						self.NE_gender[thing] = gender

			self.graph.add_edge(eventString, thing, weight=weight)
			self.graph.add_edge(thing, eventString, weight=weight)

				
	def find_pronoun_from_recent(self, i, inSent):
		index = self.prev
		neighbors = list()	
		while True:
			if index < 0: return None, None, None, "it"
			neighbors = [(n, d['weight']) for (n,i,d) in self.graph.edges(data=True) if i =="E"+str(index)]
			names = []
			weights = []
			for (n,w) in neighbors:
				names.append(n)
				weights.append(w)
			if not names: return None, None, None, "it"
			weights = [w/sum(weights) for w in weights]
			att = nx.get_node_attributes(self.graph, "att")
			n = None
			try:
				while not n or n in inSent or names:
					n = random.choice(names,1,p=weights)[0]
					i = names.index(n)
					names.remove(n)
					del weights[i]
					#n = neighbors.pop() #because it was already shuffled
			except ValueError:
				return None, None, None, "it" 
			pronoun = ""
			gender = None
			if att[n] == "<PERSON>":				
				try:
					self.getTagFromName(n,att[n])
					gender = self.NE_gender[n]
					if gender == "M":
						if i == 0: #at beginning of sentence
							pronoun = "he"
						else:
							pronoun = "him"
					elif gender == "F":
						if i == 0: #at beginning of sentence
							pronoun = "she"
						else:
							pronoun = "her"
					else:
						gender = "X"
						if i == 0: #at beginning of sentence
							pronoun = "they"
						else:
							pronoun = "them"
				except:
					gender = "X"
					if i == 0: #at beginning of sentence
						pronoun = "they"
					else:
						pronoun = "them"
			elif att[n] == "<ORGANIZATION>":
				if i == 0: #at beginning of sentence
					pronoun = "they"
				else:
					pronoun = "them"
			else:
				pronoun = "it"
			return n, att[n], gender, pronoun


	def find_recent_mentioned_item(self, category):
		att = nx.get_node_attributes(self.graph, "att")
		index = self.prev
		neighbors = list()	
		while True:
			if index < 0: return None
			neighbors = list(set(nx.all_neighbors(self.graph, "E"+str(index))))
			random.shuffle(neighbors)			
			for n in neighbors:
				try:
					if att[n] == category:
						return n
					if "Synset" in att[n] and "Synset" in category:
						_, natt, _ = att[n].split("'")
						nsyn = wn.synset(natt)
						_, cat, _ = category.split("'")
						csyn = wn.synset(cat)#[cat_num]
						lowest = nsyn.lowest_common_hypernyms(csyn)
						if category in str(lowest):
							return n
				except KeyError:
					continue

			index-=1 #go back a state and look there
		return None
