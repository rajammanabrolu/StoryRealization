# -*- coding: utf-8 -*-
#python3
import networkx as nx
import random
from nltk.corpus import wordnet as wn
import pickle

pick = open("names.pkl", 'rb')
name_list = pickle.load(pick) #name_list[name] = set(genders)
pick.close()

class MemoryGraph:
	def __init__(self):
		""" Creating a new event node that keeps track of the previous event and pointers to all things mentioned in this timestep
		"""
		self.graph = nx.DiGraph()
		self.graph.add_node("-")
		self.prev = -1

	def add_event(self, verb, thingsMentioned):
		self.prev+=1
		#create a new event node
		eventString = "E"+str(self.prev)
		newEvent = self.graph.add_node(eventString, verb=verb)
#		self.graph.add_edge(eventString, self.prev)
		#connect all recently-mentioned things to this node
		for (thing, what) in thingsMentioned:
			if not thing in self.graph.nodes():
				self.graph.add_node(thing, att=what)
			self.graph.add_edge(eventString, thing)
			self.graph.add_edge(thing, eventString)	



	def fillPronoun(gender, firstNoun, possessive):
		if gender == "M":
			if firstNoun: #at beginning of sentence
				pronoun = "he"
			elif possessive:
				pronoun = "his"
			else:
				pronoun = "him"
		elif gender == "F":
			if firstNoun: #at beginning of sentence
				pronoun = "she"
			else:
				pronoun = "her"
		else:
			gender = "X"
			if firstNoun: #at beginning of sentence
				pronoun = "they"
			elif possessive:
				pronoun = "their"
			else:
				pronoun = "them"
		return pronoun

				
	def find_pronoun_from_recent(self, NElist, firstNoun, inSent, possessive, singular):
	#	print(NElist)
		#NElist[Pearl] = (<PERSON>0,F)
		index = self.prev
		neighbors = list()	
		while True:
			if index < 0:
				if possessive: return None, None, None, "its"
				return None, None, None, "it"
			neighbors = list(set(nx.all_neighbors(self.graph, "E"+str(index))))
			random.shuffle(neighbors)
			att = nx.get_node_attributes(self.graph, "att")
			n = ""
			while ((not n) or (n in inSent) or (".a." in n)) and neighbors:
				n = neighbors.pop() #because it was already shuffled
			print("SELECTED", n)
			if not n:
				if possessive: return None, None, None, "its"
				return None, None, None, "it"
			pronoun = ""
			gender = None
			if att[n] == "<PERSON>":				
				try:
					gender = NElist[n][1]
					if gender == "M":
						if firstNoun: #at beginning of sentence
							pronoun = "he"
						elif possessive:
							pronoun = "his"
						else:
							pronoun = "him"
					elif gender == "F":
						if firstNoun: #at beginning of sentence
							pronoun = "she"
						else:
							pronoun = "her"
					else:
						gender = "X"
						if firstNoun: #at beginning of sentence
							pronoun = "they"
						elif possessive:
							pronoun = "their"
						else:
							pronoun = "them"
				except:
					gender = "X"
					if singular:
						for name in n.split(" "):
							if name in name_list:
								gender = random.choice(name_list[name])
								break
							else:
								gender = random.choice(["M","F"])
						pronoun = fillPronoun(gender, firstNoun, possessive)
					else:
						if firstNoun: #at beginning of sentence
							pronoun = "they"
						elif possessive:
							pronoun = "their"
						else:
							pronoun = "them"
			elif att[n] == "<ORGANIZATION>":
				gender = None
				if firstNoun: #at beginning of sentence
					pronoun = "they"
				elif possessive:
					pronoun = "their"
				else:
					pronoun = "them"
			else:
				gender = None
				if possessive: pronoun = "its"
				else: pronoun = "it"
			print(pronoun)
			return n, att[n], gender, pronoun


	def find_recent_mentioned_item(self, category, inSent):
		#print(category)
		att = nx.get_node_attributes(self.graph, "att")
		#print(att)
		index = self.prev
		neighbors = list()	
		while True:
			if index < 0: return None
			neighbors = list(set(nx.all_neighbors(self.graph, "E"+str(index))))
			#print(neighbors)
			random.shuffle(neighbors)			
			for n in neighbors:
				if n in inSent: continue
				if att[n] == category:
					return n
				if "Synset" in att[n] and "Synset" in category:
					_, natt, _ = att[n].split("'")
					nsyn = wn.synset(natt)
					_, cat, _ = category.split("'")
					csyn = wn.synset(cat)#[cat_num]
					#print("ATTS: ",nsyn)
					#print("CATS: ",csyn)
					lowest = nsyn.lowest_common_hypernyms(csyn)
					#print(lowest)
					if category in str(lowest):
						return n

			index-=1 #go back a state and look there
		return None
