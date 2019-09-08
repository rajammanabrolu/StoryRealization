# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from __future__ import print_function
import argparse, os, copy, random, subprocess, pickle, re, en, string
#from getEventandSpecificsPersistence_scifi import digestSentence
from nltk.corpus import wordnet as wn
from collections import defaultdict 
#import decode_e2s
from memoryGraph_scifi import MemoryGraph
import nltk.corpus
from lm import LM


verbnet = nltk.corpus.VerbnetCorpusReader('tools/VerbNet3-edit', ['absorb-39.8.xml','accept-77.1.xml','accompany-51.7.xml','acquiesce-95.1.xml','act-114.xml','addict-96.xml','adjust-26.9.xml','admire-31.2.xml','admit-64.3.xml','adopt-93.xml','advise-37.9.xml','allow-64.1.xml','amalgamate-22.2.xml','amuse-31.1.xml','animal_sounds-38.xml','appeal-31.4.xml','appear-48.1.1.xml','appoint-29.1.xml','assessment-34.1.xml','assuming_position-50.xml','attend-107.4.xml','avoid-52.xml','banish-10.2.xml','base-97.1.xml','battle-36.4.xml','become-109.1.xml','beg-58.2.xml','begin-55.1.xml','being_dressed-41.3.3.xml','bend-45.2.xml','benefit-72.2.xml','berry-13.7.xml','bill-54.5.xml','birth-28.2.xml','body_internal_motion-49.1.xml','body_internal_states-40.6.xml','body_motion-49.2.xml','braid-41.2.2.xml','break-45.1.xml','break_down-45.8.xml','breathe-40.1.2.xml','bring-11.3.xml','build-26.1.xml','bulge-47.5.3.xml','bully-59.5.xml','bump-18.4.xml','butter-9.9.xml','calibratable_cos-45.6.1.xml','calve-28.1.xml','captain-29.8.xml','care-88.1.xml','caring-75.2.xml','carry-11.4.xml','carve-21.2.xml','caused_calibratable_cos-45.6.2.xml','change_bodily_state-40.8.4.xml','characterize-29.2.xml','chase-51.6.xml','cheat-10.6.1.xml','chew-39.2.xml','chit_chat-37.6.xml','classify-29.10.xml','clear-10.3.xml','cling-22.5.xml','cognize-85.xml','coil-9.6.xml','coloring-24.xml','compel-59.1.xml','complain-37.8.xml','complete-55.2.xml','comprehend-87.2.xml','comprise-107.2.xml','concealment-16.xml','conduct-111.1.xml','confess-37.10.xml','confine-92.xml','confront-98.xml','conjecture-29.5.xml','consider-29.9.xml','conspire-71.xml','consume-66.xml','contain-15.4.xml','contiguous_location-47.8.xml','continue-55.3.xml','contribute-13.2.xml','convert-26.6.2.xml','cooking-45.3.xml','cooperate-73.1.xml','cope-83.xml','correlate-86.1.xml','correspond-36.1.1.xml','cost-54.2.xml','crane-40.3.2.xml','create-26.4.xml','curtsey-40.3.3.xml','cut-21.1.xml','debone-10.8.xml','declare-29.4.xml','dedicate-79.xml','deduce-97.2.xml','defend-72.3.xml','deprive-10.6.2.xml','destroy-44.xml','devour-39.4.xml','die-42.4.xml','differ-23.4.xml','dine-39.5.xml','disappearance-48.2.xml','disassemble-23.3.xml','discover-84.xml','disfunction-105.2.2.xml','distinguish-23.5.xml','dress-41.1.1.xml','dressing_well-41.3.2.xml','drive-11.5.xml','dub-29.3.xml','earn-54.6.xml','eat-39.1.xml','empathize-88.2.xml','employment-95.3.xml','encounter-30.5.xml','enforce-63.xml','engender-27.1.xml','ensure-99.xml','entity_specific_cos-45.5.xml','entity_specific_modes_being-47.2.xml','equip-13.4.2.xml','escape-51.1.xml','establish-55.5.xml','estimate-34.2.xml','exceed-90.xml','exchange-13.6.1.xml','exclude-107.3.xml','exhale-40.1.3.xml','exist-47.1.xml','feeding-39.7.xml','ferret-35.6.xml','fill-9.8.xml','fire-10.10.xml','fit-54.3.xml','flinch-40.5.xml','floss-41.2.1.xml','focus-87.1.xml','forbid-64.4.xml','free-10.6.3.xml','free-80.xml','fulfilling-13.4.1.xml','function-105.2.1.xml','funnel-9.3.xml','future_having-13.3.xml','get-13.5.1.xml','give-13.1.xml','gobble-39.3.xml','gorge-39.6.xml','groom-41.1.2.xml','grow-26.2.1.xml','harmonize-22.6.xml','help-72.1.xml','herd-47.5.2.xml','hiccup-40.1.1.xml','hire-13.5.3.xml','hit-18.1.xml','hold-15.1.xml','hunt-35.1.xml','hurt-40.8.3.xml','illustrate-25.3.xml','image_impression-25.1.xml','indicate-78.xml','initiate_communication-37.4.2.xml','inquire-37.1.2.xml','instr_communication-37.4.1.xml','intend-61.2.xml','interact-36.6.xml','interrogate-37.1.3.xml','invest-13.5.4.xml','investigate-35.4.xml','involve-107.1.xml','judgment-33.1.xml','keep-15.2.xml','knead-26.5.xml','learn-14.xml','leave-51.2.xml','lecture-37.11.xml','let-64.2.xml','light_emission-43.1.xml','limit-76.xml','linger-53.1.xml','lodge-46.xml','long-32.2.xml','lure-59.3.xml','manner_speaking-37.3.xml','marry-36.2.xml','marvel-31.3.xml','masquerade-29.6.xml','matter-91.xml','meander-47.7.xml','meet-36.3.xml','mine-10.9.xml','mix-22.1.xml','modes_of_being_with_motion-47.3.xml','multiply-108.xml','murder-42.1.xml','neglect-75.1.xml','nonvehicle-51.4.2.xml','nonverbal_expression-40.2.xml','obtain-13.5.2.xml','occur-48.3.xml','order-58.3.xml','orphan-29.7.xml','other_cos-45.4.xml','overstate-37.12.xml','own-100.1.xml','pain-40.8.1.xml','patent-101.xml','pay-68.xml','peer-30.3.xml','pelt-17.2.xml','performance-26.7.xml','pit-10.7.xml','pocket-9.10.xml','poison-42.2.xml','poke-19.xml','pour-9.5.xml','preparing-26.3.xml','price-54.4.xml','promise-37.13.xml','promote-102.xml','pronounce-29.3.1.xml','prosecute-33.2.xml','push-12.xml','put-9.1.xml','put_direction-9.4.xml','put_spatial-9.2.xml','reach-51.8.xml','rear-26.2.2.xml','reciprocate-112.xml','reflexive_appearance-48.1.2.xml','refrain-69.xml','register-54.1.xml','rehearse-26.8.xml','reject-77.2.xml','relate-86.2.xml','rely-70.xml','remedy-45.7.xml','remove-10.1.xml','render-29.90.xml','representation-110.1.xml','require-103.xml','resign-10.11.xml','respond-113.xml','result-27.2.xml','risk-94.xml','rob-10.6.4.xml','roll-51.3.1.xml','rummage-35.5.xml','run-51.3.2.xml','rush-53.2.xml','satisfy-55.7.xml','say-37.7.xml','scribble-25.2.xml','search-35.2.xml','see-30.1.xml','seem-109.xml','send-11.1.xml','separate-23.1.xml','settle-36.1.2.xml','shake-22.3.xml','sight-30.2.xml','simple_dressing-41.3.1.xml','slide-11.2.xml','smell_emission-43.3.xml','snooze-40.4.xml','sound_emission-43.2.xml','sound_existence-47.4.xml','spank-18.3.xml','spatial_configuration-47.6.xml','spend_time-104.xml','split-23.2.xml','spray-9.7.xml','stalk-35.3.xml','steal-10.5.xml','stimulate-59.4.xml','stimulus_subject-30.4.xml','stop-55.4.xml','subjugate-42.3.xml','subordinate-95.2.1.xml','substance_emission-43.4.xml','substitute-13.6.2.xml','succeed-74.xml','suffocate-40.7.xml','supervision-95.2.2.xml','support-15.3.xml','suspect-81.xml','sustain-55.6.xml','swarm-47.5.1.xml','swat-18.2.xml','talk-37.5.xml','tape-22.4.xml','tell-37.2.xml','terminus-47.9.xml','throw-17.1.xml','tingle-40.8.2.xml','touch-20.xml','transcribe-25.4.xml','transfer_mesg-37.1.1.xml','trick-59.2.xml','trifle-105.3.xml','try-61.1.xml','turn-26.6.1.xml','urge-58.1.xml','use-105.1.xml','vehicle-51.4.1.xml','vehicle_path-51.4.3.xml','vn_class-3.dtd','vn_schema-3.xsd','void-106.xml','volunteer-95.4.xml','waltz-51.5.xml','want-32.1.xml','weather-57.xml','weekend-56.xml','wink-40.3.1.xml','wipe_instr-10.4.2.xml','wipe_manner-10.4.1.xml','wish-62.xml','withdraw-82.xml','work-73.2.xml'])

class FillIn:
	def __init__(self):
		self.LM = LM()
		self.memory = MemoryGraph()
		self.NElist = defaultdict(tuple) #NElist[Pearl] = (<PERSON>0,F)
		self.NEnums = defaultdict(str) #NElist[<PERSON>0] = Pearl

		pick = open("names.pkl", 'rb')
		self.name_list = pickle.load(pick) #name_list[name] = set(genders)
		pick.close()

		NER_file = [x.split(";") for x in open("sci-fi_ner-lowercase-noPeriod-namechange.txt", 'r').readlines()]
		NER_dict = {k.strip(): ("<MISC>" if v.strip()=="O" else "<"+v.strip()+">") for k,v in NER_file} #entity: tag
		del(NER_file)
		self.NER_dict = NER_dict
		self.rev_NER_dict = defaultdict(set) #tag: set(entities)
		for key, value in NER_dict.items():
			if "LOCATION" in value or "PERSON" in value or "ORGANIZATION" in value or "MISC" in value:
				self.rev_NER_dict[value].add(key.title())
			else:
				self.rev_NER_dict[value].add(key)
	def reset(self):
                self.LM = LM()
                self.memory = MemoryGraph()
                self.NElist = defaultdict(tuple) #NElist[Pearl] = (<PERSON>0,F)
                self.NEnums = defaultdict(str) #NElist[<PERSON>0] = Pearl



	"""
	def pickRandoName(nameList):
		while True:
			name_select = random.choice(name_list.keys())
			if name_select not in nameList: return name_select
	"""


	def findFreeNENumber(self, category):
		i = 0
		while True:
			if category+str(i) in self.NEnums:
				i+=1
			else:
				return i
		return False


	def checkIfName(self, synset):
		pos_name = ""
		if synset.startswith("Synset"):
			pos_name = synset.split(".")[0].split("'")[1]
		else:
			pos_name = synset
		pos_name = pos_name[0].upper()+pos_name[1:]
		first_name = pos_name.split(" ")[0]
		if first_name in self.name_list:
			gender = random.choice(list(self.name_list[first_name]))
			return pos_name, gender
		return None, None


	def checkIfSciFi(self, word):
		print(word)
		if word in self.NER_dict:
			return self.NER_dict[word]
		return None


	def pickNE(self,category):
		while True:
			name_select = random.choice(list(self.rev_NER_dict[category]))
			if name_select not in self.name_list:
				gender = ""
				for n in name_select.split(" "):
					if n in self.name_list:
						gender = random.choice(list(self.name_list[n]))
						break
				if not gender: gender = "X"
				return name_select, gender


	def pickVerb(self, verb, sent, numVerbs):
		#print(verb)
		if sent: prev = sent[-1]
		else: prev = ""
		vnclass = ""
		try:
			vnclass = verbnet.vnclass(verb)
		except ValueError:
			return	None
		members = list(vnclass.findall('MEMBERS/MEMBER'))
		random.shuffle(members)
		member = ""
		while member == "" and members:
			member = members.pop()
		if member =="":
			#print("RETURNING",verb.split("-")[0])
			return verb.split("-")[0]
		m_name = member.attrib["name"]
		if "_" in m_name:
			m_name = m_name.split("_")
			if numVerbs > 0: return " ".join(m_name)
			v=""
			new_m_name=[]
			for i, word in enumerate(m_name):
				if i == 0:
					try:
						if prev == "they": v = en.verb.past(word, person=3)
						else: v = en.verb.past(word, person=1)
						if v: new_m_name.append(v)
						else: new_m_name.append(word)
					except:
						new_m_name.append(word)
				else:
					new_m_name.append(word)
			return " ".join(new_m_name)
		else:
			if numVerbs > 0: return m_name
			try:
				if prev == "they": return en.verb.past(m_name, person=3 )
				return en.verb.past(m_name, person=1)
			except:
				if m_name[-1] == "e": return m_name+"d"
				else: return m_name+"ed"

			
	def fillIn(self, emptySent, memory):
		self.memory = memory
		#NElist[Pearl] = (<PERSON>0,F)
		#NEnums[<PERSON>0] = Pearl
		event_nouns = []
		filledSent = []
		verb = ""
		emptySent_list = emptySent.split(" ")
		numVerbs = 0
		firstNoun = True #if first noun then make <PRP> the subject
		#sawComma = False

		for i, category in enumerate(emptySent_list):
			category = category.strip()

			#get rid of yuck
			if category == "EmptyParameter":
				n = self.findFreeNENumber("<MISC>")
				cat = "<MISC>"
				category = cat+str(n)

			#punctuation catch
			if numVerbs >0 and category == ",":
			#if there is a verb in the previous clause then subsequent nouns should be treated as subjects
				firstNoun = True
				#sawComma = True
				numVerbs = 0
			if all(c in string.punctuation for c in category):
				filledSent.append(category)
				continue

			#it's a pronoun, figure out what type you need
			if "<PRP>" in category:
				pronoun = ""
				possessive = False
				singular = False
				if len(emptySent_list)>i+1:
					nextWord = emptySent_list[i+1]
					if nextWord == "are":
						pronoun = "they"
					if nextWord.startswith("<") or nextWord.startswith("Synset"):
					#if next word is noun or adj then make pronoun possessive
						possessive = True
					try:
						v = en.verb.present(nextWord, person="2")
						if v == nextWord: singular = True
					except:
						pass
				recent_noun, tag, gender, newpronoun = self.memory.find_pronoun_from_recent(self.NElist,firstNoun,filledSent,possessive,singular)
				if newpronoun == "it" and possessive: newpronoun = "its"
				if not pronoun: pronoun = newpronoun
				if recent_noun:
				#this is still run so that even if the pronoun is "they", the nouns it was referring to were logged
					if recent_noun not in self.NElist:			
						self.NElist[recent_noun] = (tag, gender)
						self.NEnums[tag] = recent_noun
					event_nouns.append((recent_noun, tag))
				filledSent.append(pronoun)
				firstNoun = False

			#named entity
			elif "<" in category:
				cat = category.split(">")[0]+">"			
				if category in self.NEnums: #if this <NE> is in the dictionary already, use the same name
					filledSent.append(self.NEnums[category])
					event_nouns.append((self.NEnums[category], cat))
				else: #not used this tag before
					if category.endswith(">"):
						n = self.findFreeNENumber("<MISC>")
						cat = "<MISC>"
						category = cat+str(n)
					name, gender = self.pickNE(cat)
					self.NElist[name] = (category,gender)
					self.NEnums[category] = name
					filledSent.append(name)
					event_nouns.append((name, cat))
				firstNoun = False

			#noun synset
			elif "Synset(" in category and ".n." in category: #it's a noun synset
				name, gender = self.checkIfName(category)
				cat = "<PERSON>"
				if name: #synseted by accident
					newNum = self.findFreeNENumber("<PERSON>")
					category = cat+str(newNum)
					self.NElist[name] = (category,gender)
					self.NEnums[category] = name
					filledSent.append(name)
					event_nouns.append((name, cat))
					continue
				word = self.memory.find_recent_mentioned_item(category,filledSent)
				if not word:
					word = self.LM.pickNoun(category,filledSent)
				if len(emptySent_list)>i+1:
					if emptySent_list[i+1] == "are":
						word +="s"
				filledSent.append(word)
				event_nouns.append((word, category))
				firstNoun = False

			#adjective synset
			elif "Synset(" in category and ".a." in category: #it's an adjective synset
				word = self.memory.find_recent_mentioned_item(category,filledSent)
				if word:
					filledSent.append(word)
					event_nouns.append((word, category))
				else:
					picked = self.LM.pickNoun(category,filledSent)
					filledSent.append(picked)
					event_nouns.append((picked, category))
			#possible verb
			elif "-" in category:
				alpha, num = category.split("-",1)
				picked = ""
				if alpha[-1].isalpha() and num[0].isdigit():
					picked = self.pickVerb(category, filledSent, numVerbs)
					word = ""
					if picked: word = picked
					else: word = category.upper() #it's probably a robot
					if i == len(emptySent_list)-1: filledSent.append("was")
					filledSent.append(word)
					verb = word
					numVerbs+=1				
				else:
					cat = self.checkIfSciFi(category.replace("-"," "))
					if cat and (cat == "<PERSON>" or cat == "<ORGANIZATION>" or cat == "<MISC>"):
						name = category.title()
						newNum = self.findFreeNENumber(cat)
						category = cat+str(newNum)					
						self.NElist[name] = (category,"X")
						self.NEnums[category] = name
						filledSent.append(name)
						event_nouns.append((name, cat))
						firstNoun = False
					else:
						filledSent.append(category)
						verb = category
						numVerbs+=1
				
			#word not to be filled
			else:
				cat = self.checkIfSciFi(category)
				if cat:
					name = category.capitalize()
					newNum = self.findFreeNENumber(cat)
					category = cat+str(newNum)					
					self.NElist[name] = (category,"X")
					self.NEnums[category] = name
					filledSent.append(name)
					event_nouns.append((name, cat))
				else:
					filledSent.append(category)

		#print("EVENT_NOUNS",event_nouns)
		self.memory.add_event(verb,event_nouns)
		final_sent = " ".join(filledSent)
		if final_sent[-1] != ".": final_sent+= "."
		final_sent = final_sent[0].upper() + final_sent[1:]
		print(self.NElist)
		return final_sent


