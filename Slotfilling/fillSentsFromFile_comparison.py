from memoryGraph_scifi import MemoryGraph
from fillIn_class import FillIn

vanillaIn = [x.strip() for x in open("VANILLA_output.txt",'r').readlines()]
ensembleIn = [x.strip() for x in open("FINAL_output.txt",'r').readlines()]
vanillaOut = open("VANILLA_output_filled.txt",'w')
ensembleOut = open("FINAL_output_filled.txt",'w')

memory = MemoryGraph()
fillIn_obj = FillIn()


def getAgentTurn(line, memory):
	global fillIn_obj
	if line[-1] == ".":
		k = line.rfind(".")
		line = line[:k] + " ." + line[k+1:] #can't space all of them because of Synsets
		line = line.replace(",", " , ")
	agentTurn = fillIn_obj.fillIn(line, memory)
	while "  " in agentTurn:
		agentTurn = agentTurn.replace("  ", " ")
	return agentTurn



for i, line in enumerate(vanillaIn):
	if "--------" in line:
		memory = MemoryGraph()
		fillIn_obj.reset()
		vanillaOut.write("-----------------------------------------------\n")
		ensembleOut.write("-----------------------------------------------\n")
	else:
		vanillaLine = getAgentTurn(line, memory)
		memory = fillIn_obj.memory
		vanillaOut.write(line+"|||"+vanillaLine+"\n")

		ensembleLine = getAgentTurn(ensembleIn[i], memory)		
		memory = fillIn_obj.memory
		ensembleOut.write(ensembleIn[i]+"|||"+ensembleLine+"\n")
		#memory.add_event(event2[1],agent_nouns)

vanillaOut.close()
ensembleOut.close()
