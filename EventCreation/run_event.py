 # -*- coding: utf-8 -*-
#!/usr/bin/env python3
import pickle
from eventmaker_singleSent import eventMaker
from memoryGraph_scifi import MemoryGraph

pick = open("data/names-percentage.pkl", 'rb') #US census data
gender_list = pickle.load(pick) #gender_list[name] = set(genders)
pick.close()
del(pick)

memory = MemoryGraph(gender_list)

while(True):
	print("YOU >> ",end="")
	next_event = str(input())
	if "<quit>" in next_event:
		quit()
	else:

		if next_event[-1] != ".":
			next_event+="."
		
		try:
			eventifier = eventMaker(next_event, gender_list, memory)
			event, gen_event,ns = eventifier.getEvent()
			print("Event: "+" ".join(event))
			print("Generalized Event: "+" ".join(gen_event))
		except :
			print("Sorry, I can't find an event in that sentence. Can you try another sentence?")
			continue

		memory.add_event(event, gen_event, ns)
		print("Named Entity Dictionary: "+str(memory.NEnums))



