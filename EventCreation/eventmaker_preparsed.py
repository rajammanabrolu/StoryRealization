# Extract 5-tuple events from pre-parsed sentences
from nltk.tree import Tree
import os, copy
from collections import defaultdict
import nltk.corpus
from nltk.corpus import wordnet as wn
import regex as re
import json
from numpy.random import choice
from nltk.wsd import lesk
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('parse_file', type=str, metavar="-parse",default="corpus-parsed.json")
parser.add_argument('sentence_file',type=str,metavar="-sent",default="corpus-sentences.txt")
parser.add_argument('event_file',type=str,metavar="-out",default="events.txt")
args = parser.parse_args()

# necessary parameters
verbnet = nltk.corpus.VerbnetCorpusReader('./tools/v3.3',
['absorb-39.8.xml','accept-77.1.xml','accompany-51.7.xml','acquiesce-95.1.xml','act-114.xml','addict-96.xml','adjust-26.9.xml','admire-31.2.xml','admit-64.3.xml','adopt-93.xml','advise-37.9.xml','allow-64.1.xml','amalgamate-22.2.xml','amuse-31.1.xml','animal_sounds-38.xml','appeal-31.4.xml','appear-48.1.1.xml','appoint-29.1.xml','assessment-34.1.xml','assuming_position-50.xml','attend-107.4.xml','avoid-52.xml','banish-10.2.xml','base-97.1.xml','battle-36.4.xml','become-109.1.xml','beg-58.2.xml','begin-55.1.xml','being_dressed-41.3.3.xml','bend-45.2.xml','benefit-72.2.xml','berry-13.7.xml','bill-54.5.xml','birth-28.2.xml','body_internal_motion-49.1.xml','body_internal_states-40.6.xml','body_motion-49.2.xml','braid-41.2.2.xml','break-45.1.xml','break_down-45.8.xml','breathe-40.1.2.xml','bring-11.3.xml','build-26.1.xml','bulge-47.5.3.xml','bully-59.5.xml','bump-18.4.xml','butter-9.9.xml','calibratable_cos-45.6.1.xml','calve-28.1.xml','captain-29.8.xml','care-88.1.xml','caring-75.2.xml','carry-11.4.xml','carve-21.2.xml','caused_calibratable_cos-45.6.2.xml','change_bodily_state-40.8.4.xml','characterize-29.2.xml','chase-51.6.xml','cheat-10.6.1.xml','chew-39.2.xml','chit_chat-37.6.xml','classify-29.10.xml','clear-10.3.xml','cling-22.5.xml','cognize-85.xml','coil-9.6.xml','coloring-24.xml','compel-59.1.xml','complain-37.8.xml','complete-55.2.xml','comprehend-87.2.xml','comprise-107.2.xml','concealment-16.xml','conduct-111.1.xml','confess-37.10.xml','confine-92.xml','confront-98.xml','conjecture-29.5.xml','consider-29.9.xml','conspire-71.xml','consume-66.xml','contain-15.4.xml','contiguous_location-47.8.xml','continue-55.3.xml','contribute-13.2.xml','convert-26.6.2.xml','cooking-45.3.xml','cooperate-73.1.xml','cope-83.xml','correlate-86.1.xml','correspond-36.1.1.xml','cost-54.2.xml','crane-40.3.2.xml','create-26.4.xml','curtsey-40.3.3.xml','cut-21.1.xml','debone-10.8.xml','declare-29.4.xml','dedicate-79.xml','deduce-97.2.xml','defend-72.3.xml','deprive-10.6.2.xml','destroy-44.xml','devour-39.4.xml','die-42.4.xml','differ-23.4.xml','dine-39.5.xml','disappearance-48.2.xml','disassemble-23.3.xml','discover-84.xml','disfunction-105.2.2.xml','distinguish-23.5.xml','dress-41.1.1.xml','dressing_well-41.3.2.xml','drive-11.5.xml','dub-29.3.xml','earn-54.6.xml','eat-39.1.xml','empathize-88.2.xml','employment-95.3.xml','encounter-30.5.xml','enforce-63.xml','engender-27.1.xml','ensure-99.xml','entity_specific_cos-45.5.xml','entity_specific_modes_being-47.2.xml','equip-13.4.2.xml','escape-51.1.xml','establish-55.5.xml','estimate-34.2.xml','exceed-90.xml','exchange-13.6.1.xml','exclude-107.3.xml','exhale-40.1.3.xml','exist-47.1.xml','feeding-39.7.xml','ferret-35.6.xml','fill-9.8.xml','fire-10.10.xml','fit-54.3.xml','flinch-40.5.xml','floss-41.2.1.xml','focus-87.1.xml','forbid-64.4.xml','free-10.6.3.xml','free-80.xml','fulfilling-13.4.1.xml','function-105.2.1.xml','funnel-9.3.xml','future_having-13.3.xml','get-13.5.1.xml','give-13.1.xml','gobble-39.3.xml','gorge-39.6.xml','groom-41.1.2.xml','grow-26.2.1.xml','harmonize-22.6.xml','help-72.1.xml','herd-47.5.2.xml','hiccup-40.1.1.xml','hire-13.5.3.xml','hit-18.1.xml','hold-15.1.xml','hunt-35.1.xml','hurt-40.8.3.xml','illustrate-25.3.xml','image_impression-25.1.xml','indicate-78.xml','initiate_communication-37.4.2.xml','inquire-37.1.2.xml','instr_communication-37.4.1.xml','intend-61.2.xml','interact-36.6.xml','interrogate-37.1.3.xml','invest-13.5.4.xml','investigate-35.4.xml','involve-107.1.xml','judgment-33.1.xml','keep-15.2.xml','knead-26.5.xml','learn-14.xml','leave-51.2.xml','lecture-37.11.xml','let-64.2.xml','light_emission-43.1.xml','limit-76.xml','linger-53.1.xml','lodge-46.xml','long-32.2.xml','lure-59.3.xml','manner_speaking-37.3.xml','marry-36.2.xml','marvel-31.3.xml','masquerade-29.6.xml','matter-91.xml','meander-47.7.xml','meet-36.3.xml','mine-10.9.xml','mix-22.1.xml','modes_of_being_with_motion-47.3.xml','multiply-108.xml','murder-42.1.xml','neglect-75.1.xml','nonvehicle-51.4.2.xml','nonverbal_expression-40.2.xml','obtain-13.5.2.xml','occur-48.3.xml','order-58.3.xml','orphan-29.7.xml','other_cos-45.4.xml','overstate-37.12.xml','own-100.1.xml','pain-40.8.1.xml','patent-101.xml','pay-68.xml','peer-30.3.xml','pelt-17.2.xml','performance-26.7.xml','pit-10.7.xml','pocket-9.10.xml','poison-42.2.xml','poke-19.xml','pour-9.5.xml','preparing-26.3.xml','price-54.4.xml','promise-37.13.xml','promote-102.xml','pronounce-29.3.1.xml','prosecute-33.2.xml','push-12.xml','put-9.1.xml','put_direction-9.4.xml','put_spatial-9.2.xml','reach-51.8.xml','rear-26.2.2.xml','reciprocate-112.xml','reflexive_appearance-48.1.2.xml','refrain-69.xml','register-54.1.xml','rehearse-26.8.xml','reject-77.2.xml','relate-86.2.xml','rely-70.xml','remedy-45.7.xml','remove-10.1.xml','render-29.90.xml','representation-110.1.xml','require-103.xml','resign-10.11.xml','respond-113.xml','result-27.2.xml','risk-94.xml','rob-10.6.4.xml','roll-51.3.1.xml','rummage-35.5.xml','run-51.3.2.xml','rush-53.2.xml','satisfy-55.7.xml','say-37.7.xml','scribble-25.2.xml','search-35.2.xml','see-30.1.xml','seem-109.xml','send-11.1.xml','separate-23.1.xml','settle-36.1.2.xml','shake-22.3.xml','sight-30.2.xml','simple_dressing-41.3.1.xml','slide-11.2.xml','smell_emission-43.3.xml','snooze-40.4.xml','sound_emission-43.2.xml','sound_existence-47.4.xml','spank-18.3.xml','spatial_configuration-47.6.xml','spend_time-104.xml','split-23.2.xml','spray-9.7.xml','stalk-35.3.xml','steal-10.5.xml','stimulate-59.4.xml','stimulus_subject-30.4.xml','stop-55.4.xml','subjugate-42.3.xml','subordinate-95.2.1.xml','substance_emission-43.4.xml','substitute-13.6.2.xml','succeed-74.xml','suffocate-40.7.xml','supervision-95.2.2.xml','support-15.3.xml','suspect-81.xml','sustain-55.6.xml','swarm-47.5.1.xml','swat-18.2.xml','talk-37.5.xml','tape-22.4.xml','tell-37.2.xml','terminus-47.9.xml','throw-17.1.xml','tingle-40.8.2.xml','touch-20.xml','transcribe-25.4.xml','transfer_mesg-37.1.1.xml','trick-59.2.xml','trifle-105.3.xml','try-61.1.xml','turn-26.6.1.xml','urge-58.1.xml','use-105.1.xml','vehicle-51.4.1.xml','vehicle_path-51.4.3.xml','vn_class-3.dtd','vn_schema-3.xsd','void-106.xml','volunteer-95.4.xml','waltz-51.5.xml','want-32.1.xml','weather-57.xml','weekend-56.xml','wink-40.3.1.xml','wipe_instr-10.4.2.xml','wipe_manner-10.4.1.xml','wish-62.xml','withdraw-82.xml','work-73.2.xml'])
#file of special scifi NER and tags (VESSEL and OBJECT are added)
NER = {}

with open('data/scifiNER.txt', 'r') as f:
	for line in f:
		(key, val) = line.strip().split(";")
		if val == "O": val = "MISC"
		NER[key.lower()] = "<"+val.upper()+">"


def allXinY(word1, word2):
	list1 = word1.split(" ")
	list2 = word2.split(" ")
	for word in list1:
		if word not in list2:
			return False
	return True


#get the top-level parts of speech
def getPOSs(tree):
	poslist = []
	for subtree in tree:
		if type(subtree) == nltk.tree.Tree:
			if subtree.label() == 'PP' or subtree.label() == 'S' or subtree.label() == 'NP' or subtree.label() == 'ADVP':
				poslist+= [subtree] #add this pos subtree to the list and stop
			elif 'VB' in subtree.label():
				poslist+= ['V']
			elif subtree.label() == 'SBAR':
				if "that" in subtree.leaves():
					poslist+= ["that"]
					poslist+= getPOSs(subtree)
				elif "whether" in subtree.leaves():
					poslist+= ["whether"]
					poslist+= getPOSs(subtree)
				elif "what" in subtree.leaves():
					poslist+= ["what"]
					poslist+= getPOSs(subtree)
				elif "how" in subtree.leaves():
					poslist+= ["how"]
					poslist+= getPOSs(subtree)
			else:
				poslist+= getPOSs(subtree)
	return poslist

#taken from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(s1, s2):
	if len(s1) < len(s2):
		return levenshtein(s2, s1)
	if len(s2) == 0:
		return len(s1)
	previous_row = range(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
			deletions = current_row[j] + 1       # than s2
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row    
	return previous_row[-1]

#get the primary frame and its parts of speech
def getPrimaryFrame(frame):
	primary_frame = frame.find('DESCRIPTION').attrib['primary'] #primary syntactic description
	POS_frame = [x.split(".")[0].split("_")[0].split("-")[0] for x in primary_frame.split(" ")] #list of words, minus the descriptions after the . or _
	POS_frame = list(filter(None, POS_frame)) #get rid of the empties
	return primary_frame, POS_frame #the primary_frame might not be necessary since it's the string before the descriptions are removed

def simplifyFrame(parsed_sentence, POS):
	if len(POS) <= 1:
		return parsed_sentence, POS
	if POS[0] == "PP":
		del POS[0]
		del parsed_sentence[0]
	#remove last PP and try again
	if "PP" in POS:
		last = len(POS) - POS[::-1].index("PP") - 1
		del POS[last]
		del parsed_sentence[last]
	while POS.count("V") > 1:
		last = len(POS) - POS[::-1].index("V") - 1
		POS=POS[:last]
		parsed_sentence=parsed_sentence[:last]
	return parsed_sentence, POS

class eventMaker: 
	def __init__(self,sentence, parse, named_entities):
		self.sentence = sentence
		self.genSent = ""
		self.parse = parse
		self.nouns = defaultdict(list)
		self.verbs = defaultdict(list)
		self.events = []
		self.events_original_words = []
		self.named_entities = named_entities
		self.tokens = defaultdict(list)
		self.prep = "EmptyParameter"

	def removePunct(self):
		#remove all punctuation except periods & separate numbers
		original_sent = self.sentence.lower()
		original_sent = original_sent.replace("!", ".")
		original_sent = original_sent.replace("?", ".")
		original_sent = original_sent.replace("-", " ")
		original_sent = original_sent.replace("mr.", "mr")
		original_sent = original_sent.replace("ms.", "ms")
		original_sent = original_sent.replace("mrs.", "mrs")
		original_sent = original_sent.replace("dr.", "dr")
		original_sent = original_sent.replace("drs.", "drs")
		original_sent = original_sent.replace("st.", "st")
		original_sent = original_sent.replace("sgt.", "sgt")
		original_sent = original_sent.replace("lt.", "lt")
		original_sent = original_sent.replace("fr.", "fr")
		original_sent = original_sent.replace("pp.", "pp")
		original_sent = original_sent.replace("pg.", "pg")
		original_sent = original_sent.replace("pgs.", "pgs")
		original_sent = original_sent.replace("pps.", "pps")
		original_sent = original_sent.replace(".com", " com")
		original_sent = original_sent.replace(".net", " net")
		original_sent = original_sent.replace(".edu", " edu")
		original_sent = original_sent.replace("www.", "www ")
		original_sent = original_sent.replace("...", " ")
		original_sent = original_sent.replace("$", "$ ")
		original_sent = original_sent.replace("%", " % ")
		original_sent = re.sub('[@#^&*\(\)_\+`\-=\\/,<>:;\"\.]+', " ", original_sent)
		original_sent = original_sent.replace("'s ", " s ")
		original_sent = original_sent.replace("s' ", "s ")
		original_sent = original_sent.replace("'d ", " would ")
		original_sent = original_sent.replace("'ll ", " will ")
		original_sent = original_sent.replace("'ve ", " have ")
		original_sent = original_sent.replace(" they're ", " they are ")
		original_sent = original_sent.replace(" we're ", " we are ")
		original_sent = original_sent.replace("'t ", " t ")
		while "  " in original_sent:
			original_sent = original_sent.replace("  ", " ")
		return original_sent


	def getSkeletonParse(self):
		parse_tree = Tree.fromstring(self.parse["parse"])[0]
		parsed_sentence = getPOSs(parse_tree)
		#get the sentence's syntax
		parsed_pos = []
		for parse in parsed_sentence:
			if type(parse) is str:
				parsed_pos += [parse]
			else:
				parsed_pos += [parse.label()]
		return parsed_sentence, parsed_pos


	def findVerbFromSyntax(self, classIDs):
		#get the dependency parse, the parsed sentence (NPs, PPs, etc.), and the list of the parts of speech
		parsed_sentence, parsed_pos = self.getSkeletonParse()

		bareIDs = list(set(["-".join([cla.split("-")[0],cla.split("-")[1]]) for cla in classIDs if cla.count("-") > 1])) #get the superclass instead
		classIDs += bareIDs

		#replace PPs with their internal NPs
		for subtree in parsed_sentence:
			if type(subtree) is Tree:
				if subtree.label() == "PP":
					tree_index = parsed_sentence.index(subtree)
					for s in subtree.subtrees(filter=lambda x: x.label() == 'NP'):
						parsed_sentence[tree_index] = s
						break
		
		found = False
		while not found:
			for classID in classIDs: #loop through classIDs and look through all the syntaxes
				if found: break
			
				vnclass = verbnet.vnclass(classID) #call up the verbnet file for this class
				POS_frame = []
				preps = []
			
				#find the appropriate frame
				for frame in vnclass.findall('FRAMES/FRAME'):
					roles = [] #(predicate_role, sentence_item) pairs
					wrongPrep = False
					original_POS_string, POS_frame = getPrimaryFrame(frame)

					#remove special words (how, that, what, to be, up, whether, together, out, down, apart, why, if, when, about)
					temp_POS = POS_frame
					POS_frame = []
					for p in temp_POS:
						if p.isupper():
							POS_frame.append(p)
				
					#if the sentence matches this frame
					if levenshtein(POS_frame, parsed_pos) == 0:
						x = 0
						for POSs in frame.findall('SYNTAX'): #go through SYNTAX section in the XML
							if wrongPrep: break
							for pos in POSs: #for each POS in the frame
								if wrongPrep: break
								if pos.attrib: #not a VERB or ADV, given a value
									#for subtree in parsed_sentence, a list of subtrees that we want to look at
									for i in range(x, len(parsed_sentence)):
										if type(parsed_sentence[i]) == Tree and pos.tag == parsed_sentence[i].label(): #matching POS
											x+=1
											break
										elif pos.tag == parsed_sentence[i]: #matching a verb
											x+=1
											break
										elif pos.tag == "PREP":
											if type(parsed_sentence[x]) == str: #parsed is still on verb
												wrongPrep = True
												break
											elif pos.attrib['value'] and pos.attrib['value'][0].islower(): #if it's an actual list of prepositions
												prep = ""
												for subtree in parsed_sentence: #finds the pp for this noun
													if type(subtree) == str: #it's a verb
														continue
													if subtree.label() == "PP":
														leaves = subtree.leaves()
														if parsed_sentence[x].leaves()[0] in leaves:
															prep = leaves[0]
															self.prep = prep
															break
												if prep and prep in pos.attrib['value']: #if the preposition is in this space-delimited list
													preps += [prep]
													break
												else:
													wrongPrep = True
											else:
												break		

						if wrongPrep:
							#print("Wrong preposition. Trying another frame.")
							continue



						return classID

			if not found: #if we didn't find anything, try simplifying the sentence syntax looking again
				old_pos = parsed_pos
				parsed_sentence, parsed_pos = simplifyFrame(parsed_sentence, parsed_pos)
				if old_pos == parsed_pos:
					return ""
				continue


	def getCurrentSentenceNElist(self):
		#takes the global list of named entities, goes through them one by one, and checks if they're in the original sentence (without punctuation)
		#returns potential NE for the current sentence only
		sentence_NE_list=defaultdict(str)
		new_sent = self.removePunct()
		toRemove = set()

		for ner_word in NER:
			if " "+ner_word+" " in new_sent or new_sent.startswith(ner_word+" ") or new_sent.endswith(" "+ner_word):
				#if this NE is in the sentence
				ner_tag = NER[ner_word]
				#new_sent = new_sent.replace(ner_word,ner_tag)
				sentence_NE_list[ner_word] = ner_tag
				for already_tagged in sentence_NE_list:
					if len(ner_word) > len(already_tagged) and already_tagged in ner_word and already_tagged not in toRemove:
						#and it's the biggest NE that we can find
						toRemove.add(already_tagged)

		for remove in toRemove:
			sentence_NE_list.pop(remove, None)
		return sentence_NE_list

	def lookupNoun(self,word, pos, original_sent):
		# This is a function that supports generalize_noun function
		if word == "member": return "Synset('member.n.01')"
		if len(wn.synsets(word)) > 0:
			word1 = lesk(original_sent.split(), word, pos='n')
			try:
				hyper = lambda s: s.hypernyms()
				TREE = word1.tree(hyper, depth=6)
				temp_tree = TREE		
				for i in range(2):
					try:
						temp_tree = temp_tree[1]
					except:
						break
				result = temp_tree[0]
				return str(result)
			except AttributeError:
				return word.lower()
		else:
			return word.lower()


	def lookupAdj(self,word, pos, original_sent):
		# This is a function that supports generalize_noun function 
		if len(wn.synsets(word)) > 0:
			word1 = lesk(original_sent.split(), word, pos='a')
			try:
				hyper = lambda s: s.hypernyms()
				TREE = word1.tree(hyper, depth=6)
				temp_tree = TREE		
				for i in range(2):
					try:
						temp_tree = temp_tree[1]
					except:
						break
				result = temp_tree[0]
				return str(result)
			except AttributeError:
				return word.lower()
		else:
			return word.lower()	


	def generalize_noun(self, word, sentence_NE_list):
		# This function is to support getEvent functions. Tokens have specific format(lemma, pos, ner)
		original_sent = self.sentence
		lemma = self.tokens[word][0]
		pos = self.tokens[word][1]
		ner = "<"+self.tokens[word][2].upper()+">"
		resultString = ""
		new_sent = ""
		curr_sent_NEs = sentence_NE_list.keys()	
		word = word.lower()

		if ner in self.named_entities:
			if word in self.named_entities[ner]:
				return ner+str(self.named_entities[ner].index(word))
		for sent_NE in curr_sent_NEs:
			if word == sent_NE or word in sent_NE.split(" "): #this noun is a subset of one of the pre-tagged NEs
				ner = sentence_NE_list[sent_NE]
				for collectedNE in self.named_entities[ner]:
				#for each named entity that we already have under this tag
					if allXinY(sent_NE,collectedNE):
						resultString = ner+str(self.named_entities[ner].index(collectedNE))
						return resultString
					elif allXinY(collectedNE,sent_NE): #new name is bigger, replace it
						index = self.named_entities[ner].index(collectedNE)
						self.named_entities[ner][index] = sent_NE
						resultString = ner+str(self.named_entities[ner].index(sent_NE))
						return resultString
				#we got a new named entity!
				self.named_entities[ner].append(sent_NE) # named_entities is a list to store the names of people
				resultString = ner+str(self.named_entities[ner].index(sent_NE))
				return resultString
		if word in NER:
			ner = NER[word]
			self.named_entities[ner].append(word)
			return ner+str(self.named_entities[ner].index(word))

		if lemma == "anyone" or lemma == "anybody" or lemma == "everyone" or lemma == "someone":
			return "Synset('person.n.01')"
		if lemma == "anything" or lemma == "something":
			return "Synset('entity.n.01')"
		if "NN" in pos:			
			resultString = self.lookupNoun(lemma, pos, original_sent) # get the word's ancestor
		elif "JJ" in pos:
			resultString = self.lookupAdj(lemma, pos, original_sent)
		elif "PRP" in pos:
			resultString="<PRP>"
		elif "CC" in pos:
			if lemma == "either" or lemma == "both":
				resultString = "Synset('physical_entity.n.01')"
			else:
				return word
		else:
			resultString = word
		return resultString


	def generalize_verb(self,word,tokens):
		# This function is to support getEvent functions. //// tokens[word] = [lemma, POS, NER]
		word = tokens[word][0]
		if word == "have": return "own-100"
		classids = verbnet.classids(word)
		if len(classids) > 0:
			#return choice based on weight of number of members
			selection = self.findVerbFromSyntax(classids)
			if selection == "":			
				mems = []
				for classid in classids:
					if word in str(classid):
						return str(classid)
					vnclass = verbnet.vnclass(classid)
					num = len(list(vnclass.findall('MEMBERS/MEMBER')))
					mems.append(num)
				memcount = mems
				mems = [x/float(sum(memcount)) for x in mems]
				return str(choice(classids, 1, p=mems)[0])
			else:
				return selection
		else:
			return word


	def getGenSent(self, sentence_NE_list):
		original_sentence = self.sentence.split(" ")
		sentence_string = ""
		prev_word = ""
		for current_word in original_sentence:
			if current_word.endswith("."): current_word = current_word[:-1]
			# create events
			if "<" in prev_word and prev_word != "<PRP>":
				category, num = prev_word.split(">")
				category+=">"
				if allXinY(current_word, self.named_entities[category][int(num)]) or allXinY(self.named_entities[category][int(num)], current_word):
					continue
					
			if self.tokens[current_word]:
				result_word = self.generalize_noun(current_word, sentence_NE_list)
				if prev_word == result_word: continue
				sentence_string+= " "+result_word
				prev_word = result_word
			else:
				sentence_string+= " "+current_word
				prev_word = current_word
		return sentence_string.strip()+"."


	def getEvent(self):
	# given a parse of a sentence, extract the event(s)
		#setup parameters
		deps = self.parse["enhancedPlusPlusDependencies"]	 # retrieve the dependencies

		sentence_NE_list = self.getCurrentSentenceNElist()
		verbs = []
		subjects = []
		modifiers = []
		objects = []
		pos = {}
		prepositions = defaultdict(str) #the dependentGloss of dep "case" is the preposition //// prepositions[noun] = preposition
		pos["EmptyParameter"] = "EmptyParameter"
		chainMods = {} # chaining of mods
		index = defaultdict(list)  #for identifying part-of-speech
		index["EmptyParameter"] = -1

		tokens = defaultdict(list)
				
		for token in self.parse["tokens"]:
			# each word in the dictionary has a list of [lemma, POS, NER]
			tokens[token["word"]] = [token["lemma"], token["pos"], token["ner"]]
		self.tokens = tokens

		self.genSent = self.getGenSent(sentence_NE_list)

		# create events
		for d in deps:
			#subject
			if 'nsubj' in d["dep"] and "RB" not in tokens[d["dependentGloss"]][1]:
				if d["governorGloss"] not in verbs:
					#create new event
					if not "VB" in tokens[d["governorGloss"]][1]: continue
					verbs.append(d["governorGloss"])
					index[d["governorGloss"]] = d["governor"] #adding index
					subjects.append(d["dependentGloss"])
					index[d["dependentGloss"]] = d["dependent"] #adding index to subject
					pos[d["governorGloss"]] = tokens[d["governorGloss"]][1]
					pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
					modifiers.append('EmptyParameter')
					objects.append('EmptyParameter')
				elif d["governorGloss"] in verbs:
					if subjects[verbs.index(d["governorGloss"])] == "EmptyParameter": # if verb alrady exist 
						subjects[verbs.index(d["governorGloss"])] = d["dependentGloss"]
						index[d["dependentGloss"]] = d["dependent"]
					else:
						subjects.append(d["dependentGloss"])
						index[d["dependentGloss"]] = d["dependent"]
						verbs.append(d["governorGloss"])
						index[d["governorGloss"]] = d["governor"]
						modifiers.append('EmptyParameter')
						objects.append('EmptyParameter')
					pos[d["governorGloss"]] = tokens[d["governorGloss"]][1]
					pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
				elif d["dependentGloss"] in subjects: # one subject multiple verbs
					verbs[subjects.index(d["dependentGloss"])] = d["governorGloss"]
					index[d["governorGloss"]] = d["governor"]
					pos[d["governorGloss"]] = tokens[d["governorGloss"]][1]
					pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
			else: #check to see if we have a subject filled
				if len(subjects) >1:
					if subjects[-1] == "EmptyParameter":
						subjects[-1] = subjects[-2]
				#conjunction of verbs
				if 'conj' in d["dep"] and 'VB' in tokens[d["dependentGloss"]][1]:
					if d["dependentGloss"] not in verbs:
						verbs.append(d["dependentGloss"])
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
						index[d["dependentGloss"]] = d["dependent"]
						subjects.append('EmptyParameter')
						modifiers.append('EmptyParameter')
						objects.append('EmptyParameter')
				elif 'conj' in d["dep"] and d["governorGloss"] in objects:
					loc = objects.index(d["governorGloss"])
					match_verb = verbs[loc]
					temp_verbs = copy.deepcopy(verbs)
					for i, verb in enumerate(temp_verbs):
						if match_verb == verb:
							subjects.append(subjects[i]) 
							verbs.append(verb)
							modifiers.append('EmptyParameter')
							objects.append(d["dependentGloss"])
							index[d["dependentGloss"]] = d["dependent"]
					pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]			
				# case 1: obj
				elif 'dobj' in d["dep"] or 'xcomp' == d["dep"]:
					if d["governorGloss"] in verbs:
						#modify that object
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
						for i, verb in reversed(list(enumerate(verbs))):
							if verb == d["governorGloss"] and objects[i] == "EmptyParameter":
								objects[i] = d["dependentGloss"]
								index[d["dependentGloss"]] = d["dependent"]
					#Object is a verb
					elif d["governorGloss"] in objects:
						#modify that object
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
						for i, obj in reversed(list(enumerate(objects))):
							if obj == d["governorGloss"] and modifiers[i] == "EmptyParameter":
								modifiers[i] = d["dependentGloss"]
								index[d["dependentGloss"]] = d["dependent"]
				# case 2: nmod
				elif ('nmod' in d["dep"] or 'ccomp' in d["dep"] or 'iobj' in d["dep"] or 'dep' in d["dep"]) and 'NN' in tokens[d["dependentGloss"]][1]:
					if d["governorGloss"] in verbs:
						#modify that modifier
						for i, verb in reversed(list(enumerate(verbs))):
							if verb == d["governorGloss"] and modifiers[i] == "EmptyParameter":
								modifiers[i] = d["dependentGloss"]
								index[d["dependentGloss"]] = d["dependent"]
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
					elif d["governorGloss"] in chainMods:
						v = chainMods[d["governorGloss"]]
						if v in verbs:
							modifiers[verbs.index(v)] = d["dependentGloss"]
							index[d["dependentGloss"]] = d["dependent"]
							pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
				# PRP (pronoun)		
				elif ('nmod' in d["dep"] or 'ccomp' in d["dep"] or 'iobj' in d["dep"] or 'dep' in d["dep"]) and 'PRP' in tokens[d["dependentGloss"]][1]:
					if d["governorGloss"] in verbs:
						#modify that modifier
						for i, verb in reversed(list(enumerate(verbs))):
							if verb == d["governorGloss"] and modifiers[i] == "EmptyParameter":
								modifiers[i] = d["dependentGloss"]
								index[d["dependentGloss"]] = d["dependent"]
						pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
					elif d["governorGloss"] in chainMods:
						v = chainMods[d["governorGloss"]]
						if v in verbs:
							modifiers[verbs.index(v)] = d["dependentGloss"]
							index[d["dependentGloss"]] = d["dependent"]
							pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
				#Complex Verbs	
				elif 'acl' in d["dep"] and d["governorGloss"] in verbs:
					for i, verb in reversed(list(enumerate(verbs))):
						if verb == d["governorGloss"] and objects[i] == "EmptyParameter":
							objects[i] = d["dependentGloss"]
							index[d["dependentGloss"]] = d["dependent"]
					pos[d["dependentGloss"]] = tokens[d["dependentGloss"]][1]
				#preposition
				elif 'case' in d["dep"]:
					noun = d["governorGloss"]
					prep = d["dependentGloss"].lower()
					prepositions[noun] = prep

				
		# generalize the words and store them in instance variables
		for (a,b,c,d) in zip(subjects, verbs, objects, modifiers):
			num = 0
			if a != 'EmptyParameter':
				a1 = self.generalize_noun(a, sentence_NE_list)
				if a1 == a:
					a1 = self.generalize_verb(a, tokens)
					self.verbs[a1].append(tokens[a][0])
				else:
					self.nouns[a1].append(tokens[a][0])
			else:
				a1 = a
			if b != 'EmptyParameter':
				b1 = self.generalize_verb(b, tokens)
				self.verbs[b1].append(tokens[b][0])
			else:
				b1 = b
			if c != 'EmptyParameter':
				c1 = self.generalize_noun(c, sentence_NE_list)
				if c1 == c:
					c1 = self.generalize_verb(c, tokens)
					self.verbs[c1].append(tokens[c][0])
				else:
					self.nouns[c1].append(tokens[c][0])
			else:
				c1 = c
			if d == 'EmptyParameter':
				label = 'EmptyParameter'
			else:
				label = 'None'
				for dep in deps:
					if b == dep["governorGloss"] and d == dep["dependentGloss"] and "nmod" in dep["dep"]:
						if ":" in dep["dep"]:
							label = dep["dep"].split(":")[1]
						num = dep['dependent']
				for dep in deps:
					if b == dep["governorGloss"] and d == dep["dependentGloss"] and "obl" in dep["dep"]:
						if ":" in dep["dep"]:
							label = dep["dep"].split(":")[1]
						num = dep['dependent']
							
				for dep in deps:
					if "case" in dep["dep"] and d == dep["governorGloss"] and num == dep['governor']:
						label = dep["dependentGloss"]
						index[label] = dep["dependent"]
			if d != 'EmptyParameter':
				d1 = self.generalize_noun(d, sentence_NE_list)
			else:
				d1 = d
			returningEvent = [a1,b1,c1,d1,"EmptyParameter"]
			orig = [a,b,c,d,"EmptyParameter"]
			returningEvent_orig = [tokens[x][0] if x != "EmptyParameter" else x for x in orig]
			for i, param in enumerate(returningEvent):
				#getting rid of the "None" tag
				if param == "None":
					returningEvent[i] = "EmptyParameter"
				if returningEvent_orig[i] == "None":
					returningEvent_orig[i] = "EmptyParameter"
			#adding in prepositions
			if d in prepositions:
				returningEvent[4] = prepositions[d]
				returningEvent_orig[4] = prepositions[d]
			if returningEvent[4] == "EmptyParameter" and c in prepositions:
				returningEvent[4] = prepositions[c]
				returningEvent_orig[4] = prepositions[c]
			
			self.events_original_words.append(returningEvent_orig)
			self.events.append(returningEvent)


###################################
#SETUP
#file of parses
parsed_file = open(args.parse_file, 'r')
#list of all the sentences in the same order as the parses
sentences = [sent.strip() for sent in open(args.sentence_file, 'r').readlines()]

###################################
#RUN
parsed_json = json.load(parsed_file)
PARSES = parsed_json["parses"]
del parsed_json
numEOS = 0
named_entities = defaultdict(list)
output_file = open(args.event_file, 'w')
for i, parse in enumerate(PARSES):
	if "<EOS>" in sentences[i+numEOS]:
		output_file.write("<EOS>/n")
		output_file.write("%%%%%%%%%%%%%%%%%"+str(named_entities)+"\n")
		numEOS+=1
		named_entities = defaultdict(list) #restarts at new story
	maker = eventMaker(sentences[i+numEOS], parse["sentences"][0], named_entities)
	named_entities = maker.named_entities
	maker.getEvent()
	if maker.events:
		output_file.write(str(maker.events_original_words)+"|||"+str(maker.events)+"|||"+sentences[i+numEOS]+"|||"+maker.genSent+"\n")
output_file.write("<EOS>\n")
output_file.write("%%%%%%%%%%%%%%%%%"+str(named_entities)+"\n")
parsed_file.close()
output_file.close()

