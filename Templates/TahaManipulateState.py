# -*- coding: utf-8 -*-
import nltk.corpus
#from parseStanford import StanfordParser, StanfordDependencyParser #a "hacked" version of the NLTK parse file
import copy
from nltk.tree import Tree
import random
#from conceptnet5.query import lookup, query
from collections import defaultdict
#import wikipedia
from numpy import random
#from changePredicates import *

"""
Verbs not found in 3.3
rig
impact
BUMBLE
sabotage
account
await
apologize
harbor
incapacitate
pinpoint
bestow
unearth
locate
invade
deploy
reminisce
brandish
violate
withhold
relent
dissect
bid
reprimand #added to judgment-33.1-1
misjudge #added to conjecture-29.5-1
oversee
decloak
accost
renege
vow
coordinate
jettison #added to throw-17.1
refute
recreate
interogate  # should be interrogate
plead #created beg-58.2-2
schedule
interrupt
cancel
invoke
opt
piece
resist
share
feature
preside
comply
weather
inhabit
pressurize
dial
disrupt
calculate
forget
strand
malfunction
freak
overrule
reactivate
task
iterate
bomb #added to pelt-17.2
suitable



['<PERSON>8', 'indicate-78-1', 'EmptyParameter', 'EmptyParameter', 'EmptyParameter']
["Synset('physical_entity.n.01')", 'comprehend-87.2-1-1-1', 'EmptyParameter', 'EmptyParameter', 'EmptyParameter']
["Synset('male.n.02')", 'discover-84', 'EmptyParameter', 'EmptyParameter', 'EmptyParameter']
["Synset('female.n.02')", 'accept-77.1', 'EmptyParameter', 'EmptyParameter', 'EmptyParameter']
["Synset('male.n.02')", 'conjecture-29.5-1', 'EmptyParameter', 'EmptyParameter', 'EmptyParameter']
['sun-hawk', 'escape-51.1-1', 'EmptyParameter', 'EmptyParameter', 'EmptyParameter']
['long-shot', 'work-73.2', 'EmptyParameter', "Synset('abstraction.n.06')", 'without']
["Synset('male.n.02')", 'confess-37.10', 'EmptyParameter', 'EmptyParameter', 'EmptyParameter']
["Synset('physical_entity.n.01')", 'promise-37.13', 'EmptyParameter', 'EmptyParameter', 'EmptyParameter']
"""

#verbnet = nltk.corpus.VerbnetCorpusReader('/mnt/sdb1/Pipeline/tools/verbnet', ['absorb-39.8.xml','continue-55.3.xml','hurt-40.8.3.xml','remove-10.1.xml','accept-77.xml','contribute-13.2.xml','illustrate-25.3.xml','render-29.90.xml','accompany-51.7.xml','convert-26.6.2.xml','image_impression-25.1.xml','require-103.xml','acquiesce-95.xml','cooking-45.3.xml','indicate-78.xml','resign-10.11.xml','addict-96.xml','cooperate-73.xml','inquire-37.1.2.xml','risk-94.xml','adjust-26.9.xml','cope-83.xml','instr_communication-37.4.xml','roll-51.3.1.xml','admire-31.2.xml','correlate-86.1.xml','interrogate-37.1.3.xml','rummage-35.5.xml','admit-65.xml','correspond-36.1.xml','investigate-35.4.xml','run-51.3.2.xml','adopt-93.xml','cost-54.2.xml','involve-107.xml','rush-53.2.xml','advise-37.9.xml','crane-40.3.2.xml','judgment-33.xml','say-37.7.xml','allow-64.xml','create-26.4.xml','keep-15.2.xml','scribble-25.2.xml','amalgamate-22.2.xml','curtsey-40.3.3.xml','knead-26.5.xml','search-35.2.xml','amuse-31.1.xml','cut-21.1.xml','learn-14.xml','see-30.1.xml','animal_sounds-38.xml','debone-10.8.xml','leave-51.2.xml','seem-109.xml','appeal-31.4.xml','declare-29.4.xml','lecture-37.11.xml','send-11.1.xml','appear-48.1.1.xml','dedicate-79.xml','light_emission-43.1.xml','separate-23.1.xml','appoint-29.1.xml','deduce-97.2.xml','limit-76.xml','settle-89.xml','assessment-34.1.xml','defend-72.2.xml','linger-53.1.xml','shake-22.3.xml','assuming_position-50.xml','destroy-44.xml','sight-30.2.xml','avoid-52.xml','devour-39.4.xml','lodge-46.xml','simple_dressing-41.3.1.xml','banish-10.2.xml','differ-23.4.xml','long-32.2.xml','slide-11.2.xml','base-97.1.xml','dine-39.5.xml','manner_speaking-37.3.xml','smell_emission-43.3.xml','battle-36.4.xml','disappearance-48.2.xml','marry-36.2.xml','snooze-40.4.xml','become-109.1.xml','disassemble-23.3.xml','marvel-31.3.xml','beg-58.2.xml','discover-84.xml','masquerade-29.6.xml','sound_emission-43.2.xml','begin-55.1.xml','dress-41.1.1.xml','matter-91.xml','sound_existence-47.4.xml','being_dressed-41.3.3.xml','dressing_well-41.3.2.xml','meander-47.7.xml','spank-18.3.xml','bend-45.2.xml','drive-11.5.xml','meet-36.3.xml','spatial_configuration-47.6.xml','benefit-72.1.xml','dub-29.3.xml','mine-10.9.xml','spend_time-104.xml','berry-13.7.xml','eat-39.1.xml','mix-22.1.xml','split-23.2.xml','bill-54.5.xml','empathize-88.2.xml','modes_of_being_with_motion-47.3.xml','spray-9.7.xml','body_internal_motion-49.xml','enforce-63.xml','multiply-108.xml','stalk-35.3.xml','body_internal_states-40.6.xml','engender-27.xml','murder-42.1.xml','steal-10.5.xml','braid-41.2.2.xml','ensure-99.xml','neglect-75.xml','stimulus_subject-30.4.xml','break-45.1.xml','entity_specific_cos-45.5.xml','nonvehicle-51.4.2.xml','stop-55.4.xml','breathe-40.1.2.xml','entity_specific_modes_being-47.2.xml','nonverbal_expression-40.2.xml','subjugate-42.3.xml','bring-11.3.xml','equip-13.4.2.xml','obtain-13.5.2.xml','substance_emission-43.4.xml','build-26.1.xml','escape-51.1.xml','occurrence-48.3.xml','succeed-74.xml','bulge-47.5.3.xml','establish-55.5.xml','order-60.xml','suffocate-40.7.xml','bump-18.4.xml','estimate-34.2.xml','orphan-29.7.xml','suspect-81.xml','butter-9.9.xml','exceed-90.xml','other_cos-45.4.xml','sustain-55.6.xml','calibratable_cos-45.6.xml','exchange-13.6.xml','overstate-37.12.xml','swarm-47.5.1.xml','calve-28.xml','exhale-40.1.3.xml','own-100.xml','swat-18.2.xml','captain-29.8.xml','exist-47.1.xml','pain-40.8.1.xml','talk-37.5.xml','care-88.1.xml','feeding-39.7.xml','patent-101.xml','tape-22.4.xml','carry-11.4.xml','ferret-35.6.xml','pay-68.xml','tell-37.2.xml','carve-21.2.xml','fill-9.8.xml','peer-30.3.xml','throw-17.1.xml','change_bodily_state-40.8.4.xml','fire-10.10.xml','pelt-17.2.xml','tingle-40.8.2.xml','characterize-29.2.xml','fit-54.3.xml','performance-26.7.xml','touch-20.xml','chase-51.6.xml','flinch-40.5.xml','pit-10.7.xml','transcribe-25.4.xml','cheat-10.6.xml','floss-41.2.1.xml','pocket-9.10.xml','transfer_mesg-37.1.1.xml','chew-39.2.xml','focus-87.1.xml','poison-42.2.xml','try-61.xml','chit_chat-37.6.xml','forbid-67.xml','poke-19.xml','turn-26.6.1.xml','classify-29.10.xml','force-59.xml','pour-9.5.xml','urge-58.1.xml','clear-10.3.xml','free-80.xml','preparing-26.3.xml','use-105.xml','cling-22.5.xml','fulfilling-13.4.1.xml','price-54.4.xml','vehicle-51.4.1.xml','coil-9.6.xml','funnel-9.3.xml','promise-37.13.xml','vehicle_path-51.4.3.xml','coloring-24.xml','future_having-13.3.xml','promote-102.xml','complain-37.8.xml','get-13.5.1.xml','pronounce-29.3.1.xml','complete-55.2.xml','give-13.1.xml','push-12.xml','void-106.xml','comprehend-87.2.xml','gobble-39.3.xml','put-9.1.xml','waltz-51.5.xml','comprise-107.1.xml','gorge-39.6.xml','put_direction-9.4.xml','want-32.1.xml','concealment-16.xml','groom-41.1.2.xml','put_spatial-9.2.xml','weather-57.xml','confess-37.10.xml','grow-26.2.xml','reach-51.8.xml','weekend-56.xml','confine-92.xml','help-72.xml','reflexive_appearance-48.1.2.xml','wink-40.3.1.xml','confront-98.xml','herd-47.5.2.xml','refrain-69.xml','wipe_instr-10.4.2.xml','conjecture-29.5.xml','hiccup-40.1.1.xml','register-54.1.xml','wipe_manner-10.4.1.xml','consider-29.9.xml','hire-13.5.3.xml','rehearse-26.8.xml','wish-62.xml','conspire-71.xml','hit-18.1.xml','relate-86.2.xml','withdraw-82.xml','consume-66.xml','hold-15.1.xml','rely-70.xml','contiguous_location-47.8.xml','hunt-35.1.xml','remedy-45.7.xml'])
verbnet = nltk.corpus.VerbnetCorpusReader('/home/twister/ASTER/E2S-Ensemble/Templates/VerbNet3-edit', ['absorb-39.8.xml','accept-77.1.xml','accompany-51.7.xml','acquiesce-95.1.xml','act-114.xml','addict-96.xml','adjust-26.9.xml','admire-31.2.xml','admit-64.3.xml','adopt-93.xml','advise-37.9.xml','allow-64.1.xml','amalgamate-22.2.xml','amuse-31.1.xml','animal_sounds-38.xml','appeal-31.4.xml','appear-48.1.1.xml','appoint-29.1.xml','assessment-34.1.xml','assuming_position-50.xml','attend-107.4.xml','avoid-52.xml','banish-10.2.xml','base-97.1.xml','battle-36.4.xml','become-109.1.xml','beg-58.2.xml','begin-55.1.xml','being_dressed-41.3.3.xml','bend-45.2.xml','benefit-72.2.xml','berry-13.7.xml','bill-54.5.xml','birth-28.2.xml','body_internal_motion-49.1.xml','body_internal_states-40.6.xml','body_motion-49.2.xml','braid-41.2.2.xml','break-45.1.xml','break_down-45.8.xml','breathe-40.1.2.xml','bring-11.3.xml','build-26.1.xml','bulge-47.5.3.xml','bully-59.5.xml','bump-18.4.xml','butter-9.9.xml','calibratable_cos-45.6.1.xml','calve-28.1.xml','captain-29.8.xml','care-88.1.xml','caring-75.2.xml','carry-11.4.xml','carve-21.2.xml','caused_calibratable_cos-45.6.2.xml','change_bodily_state-40.8.4.xml','characterize-29.2.xml','chase-51.6.xml','cheat-10.6.1.xml','chew-39.2.xml','chit_chat-37.6.xml','classify-29.10.xml','clear-10.3.xml','cling-22.5.xml','cognize-85.xml','coil-9.6.xml','coloring-24.xml','compel-59.1.xml','complain-37.8.xml','complete-55.2.xml','comprehend-87.2.xml','comprise-107.2.xml','concealment-16.xml','conduct-111.1.xml','confess-37.10.xml','confine-92.xml','confront-98.xml','conjecture-29.5.xml','consider-29.9.xml','conspire-71.xml','consume-66.xml','contain-15.4.xml','contiguous_location-47.8.xml','continue-55.3.xml','contribute-13.2.xml','convert-26.6.2.xml','cooking-45.3.xml','cooperate-73.1.xml','cope-83.xml','correlate-86.1.xml','correspond-36.1.1.xml','cost-54.2.xml','crane-40.3.2.xml','create-26.4.xml','curtsey-40.3.3.xml','cut-21.1.xml','debone-10.8.xml','declare-29.4.xml','dedicate-79.xml','deduce-97.2.xml','defend-72.3.xml','deprive-10.6.2.xml','destroy-44.xml','devour-39.4.xml','die-42.4.xml','differ-23.4.xml','dine-39.5.xml','disappearance-48.2.xml','disassemble-23.3.xml','discover-84.xml','disfunction-105.2.2.xml','distinguish-23.5.xml','dress-41.1.1.xml','dressing_well-41.3.2.xml','drive-11.5.xml','dub-29.3.xml','earn-54.6.xml','eat-39.1.xml','empathize-88.2.xml','employment-95.3.xml','encounter-30.5.xml','enforce-63.xml','engender-27.1.xml','ensure-99.xml','entity_specific_cos-45.5.xml','entity_specific_modes_being-47.2.xml','equip-13.4.2.xml','escape-51.1.xml','establish-55.5.xml','estimate-34.2.xml','exceed-90.xml','exchange-13.6.1.xml','exclude-107.3.xml','exhale-40.1.3.xml','exist-47.1.xml','feeding-39.7.xml','ferret-35.6.xml','fill-9.8.xml','fire-10.10.xml','fit-54.3.xml','flinch-40.5.xml','floss-41.2.1.xml','focus-87.1.xml','forbid-64.4.xml','free-10.6.3.xml','free-80.xml','fulfilling-13.4.1.xml','function-105.2.1.xml','funnel-9.3.xml','future_having-13.3.xml','get-13.5.1.xml','give-13.1.xml','gobble-39.3.xml','gorge-39.6.xml','groom-41.1.2.xml','grow-26.2.1.xml','harmonize-22.6.xml','help-72.1.xml','herd-47.5.2.xml','hiccup-40.1.1.xml','hire-13.5.3.xml','hit-18.1.xml','hold-15.1.xml','hunt-35.1.xml','hurt-40.8.3.xml','illustrate-25.3.xml','image_impression-25.1.xml','indicate-78.xml','initiate_communication-37.4.2.xml','inquire-37.1.2.xml','instr_communication-37.4.1.xml','intend-61.2.xml','interact-36.6.xml','interrogate-37.1.3.xml','invest-13.5.4.xml','investigate-35.4.xml','involve-107.1.xml','judgment-33.1.xml','keep-15.2.xml','knead-26.5.xml','learn-14.xml','leave-51.2.xml','lecture-37.11.xml','let-64.2.xml','light_emission-43.1.xml','limit-76.xml','linger-53.1.xml','lodge-46.xml','long-32.2.xml','lure-59.3.xml','manner_speaking-37.3.xml','marry-36.2.xml','marvel-31.3.xml','masquerade-29.6.xml','matter-91.xml','meander-47.7.xml','meet-36.3.xml','mine-10.9.xml','mix-22.1.xml','modes_of_being_with_motion-47.3.xml','multiply-108.xml','murder-42.1.xml','neglect-75.1.xml','nonvehicle-51.4.2.xml','nonverbal_expression-40.2.xml','obtain-13.5.2.xml','occur-48.3.xml','order-58.3.xml','orphan-29.7.xml','other_cos-45.4.xml','overstate-37.12.xml','own-100.1.xml','pain-40.8.1.xml','patent-101.xml','pay-68.xml','peer-30.3.xml','pelt-17.2.xml','performance-26.7.xml','pit-10.7.xml','pocket-9.10.xml','poison-42.2.xml','poke-19.xml','pour-9.5.xml','preparing-26.3.xml','price-54.4.xml','promise-37.13.xml','promote-102.xml','pronounce-29.3.1.xml','prosecute-33.2.xml','push-12.xml','put-9.1.xml','put_direction-9.4.xml','put_spatial-9.2.xml','reach-51.8.xml','rear-26.2.2.xml','reciprocate-112.xml','reflexive_appearance-48.1.2.xml','refrain-69.xml','register-54.1.xml','rehearse-26.8.xml','reject-77.2.xml','relate-86.2.xml','rely-70.xml','remedy-45.7.xml','remove-10.1.xml','render-29.90.xml','representation-110.1.xml','require-103.xml','resign-10.11.xml','respond-113.xml','result-27.2.xml','risk-94.xml','rob-10.6.4.xml','roll-51.3.1.xml','rummage-35.5.xml','run-51.3.2.xml','rush-53.2.xml','satisfy-55.7.xml','say-37.7.xml','scribble-25.2.xml','search-35.2.xml','see-30.1.xml','seem-109.xml','send-11.1.xml','separate-23.1.xml','settle-36.1.2.xml','shake-22.3.xml','sight-30.2.xml','simple_dressing-41.3.1.xml','slide-11.2.xml','smell_emission-43.3.xml','snooze-40.4.xml','sound_emission-43.2.xml','sound_existence-47.4.xml','spank-18.3.xml','spatial_configuration-47.6.xml','spend_time-104.xml','split-23.2.xml','spray-9.7.xml','stalk-35.3.xml','steal-10.5.xml','stimulate-59.4.xml','stimulus_subject-30.4.xml','stop-55.4.xml','subjugate-42.3.xml','subordinate-95.2.1.xml','substance_emission-43.4.xml','substitute-13.6.2.xml','succeed-74.xml','suffocate-40.7.xml','supervision-95.2.2.xml','support-15.3.xml','suspect-81.xml','sustain-55.6.xml','swarm-47.5.1.xml','swat-18.2.xml','talk-37.5.xml','tape-22.4.xml','tell-37.2.xml','terminus-47.9.xml','throw-17.1.xml','tingle-40.8.2.xml','touch-20.xml','transcribe-25.4.xml','transfer_mesg-37.1.1.xml','trick-59.2.xml','trifle-105.3.xml','try-61.1.xml','turn-26.6.1.xml','urge-58.1.xml','use-105.1.xml','vehicle-51.4.1.xml','vehicle_path-51.4.3.xml','vn_class-3.dtd','vn_schema-3.xsd','void-106.xml','volunteer-95.4.xml','waltz-51.5.xml','want-32.1.xml','weather-57.xml','weekend-56.xml','wink-40.3.1.xml','wipe_instr-10.4.2.xml','wipe_manner-10.4.1.xml','wish-62.xml','withdraw-82.xml','work-73.2.xml'])


#uncomment and run once to download these packages
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('verbnet')
# nltk.download('propbank')

#stanford_dir = os.path.join(os.getcwd(), "tools/stanford-tools")

"""
@author: Lara Martin -- may be modified by Taha Merghani
"""

############################################
class ManipulateState:
    def __init__(self, event, NERs):#sent, facts = {}):
        """ Creating a new state manipulator
        """
        self.facts = {}
        self.currentlyMentioned = defaultdict(set)
        self.roles = defaultdict(str) #roles[predicate_role] = sentence_item
        self.NER = NERs
        self.addState(event)
        self.sels = defaultdict(set)
        #self.changePredicateObject = None

    """
    def addDictionaries(self,d1, d2):
        #TODO add in predicate rules
        temp = copy.deepcopy(d1)
        for item in temp: #put currently mentioned in newdict
            if item in d2:
                d2[item]|=temp[item]
            else:
                d2[item]=temp[item]
        return d2
    """

    #get the primary frame and its parts of speech
    def getPrimaryFrame(self,frame):
        primary_frame = frame.find('DESCRIPTION').attrib['primary'] #primary syntactic description
        POS_frame = [x.split(".")[0].split("_")[0].split("-")[0] for x in primary_frame.split(" ")] #list of words, minus the descriptions after the . or _ 
        POS_frame = list(filter(None, POS_frame)) #get rid of the empties

        #remove special words (how, that, what, to be, up, whether, together, out, down, apart, why, if, when, about)
        temp_POS = POS_frame
        POS_frame = []
        for p in temp_POS:
            if p.isupper():
                POS_frame.append(p)
        return primary_frame, POS_frame #the primary_frame might not be necessary since it's the string before the descriptions are removed

    #collect the selectional restrictions for this verb class
    def getSelectors(self,vnclass):
        #note the thematic roles in this class
        class_selectors = defaultdict(set)
        for theme in vnclass.findall('THEMROLES/THEMROLE'):
            themeType = theme.attrib['type']
            sels = theme.find('SELRESTRS') #there should only be one
            found = False
            if 'logic' in sels.attrib:                    #ORs (e.g. "interrogate")
                selSet = set()
                for sel in sels.findall('SELRESTR'):
                    found = True
                    selSet.add(sel.attrib['Value']+sel.attrib['type'])
                or_set = "|".join(selSet)
                class_selectors[themeType]|=set([or_set])
            else:                                #ANDs (e.g. "reach")
                for sel in sels.findall('SELRESTR'):
                    found = True
                    class_selectors[themeType]|=set([sel.attrib['Value']+sel.attrib['type']])
            #if not found:
            #    class_selectors[themeType]|=set()    #add it just in case
        return class_selectors

    #put everything in the right place in the predicates, including changing predicates to "core predicates"
    #make sure that this verb is legal
    def fillPredicates(self,frame, preps, currentFacts):
        #fill the predicates' roles with the items in the sentence        
        prep_count = 0
        pred_objs = []

        #go through the predicates and find any "equals" first
        for pred in frame.findall('SEMANTICS/PRED'):
            if pred.attrib['value'] == "equals":
                same_entity = ""
                empty_value = ""
                for arg in pred.findall('ARGS/ARG'): #for each argument in this predicate
                    if arg.attrib['value'] in self.roles.keys():
                        same_entity = self.roles[arg.attrib['value']]
                    else:
                        empty_value = arg.attrib['value']
                self.roles[empty_value] = same_entity
                break
        for pred in frame.findall('SEMANTICS/PRED'): #go through each predicate in this frame
            negated = False
            if 'bool' in pred.attrib.keys(): negated = True
            predicate = pred.attrib['value']+"("
            eventType = ""
            printlist = []
            list_for_pred_change = []
            for arg in pred.findall('ARGS/ARG'): #for each argument in this predicate
                if arg.attrib['type'] == "Event" or arg.attrib['type'] == "Constant":
                    eventType = arg.attrib['value']
                    printlist.append(eventType)
                    list_for_pred_change.append(eventType)
                #elif arg.attrib['type'] == 'refl':
                elif "?" in arg.attrib['value']:
                    printlist.append(arg.attrib['value'])
                    list_for_pred_change.append(arg.attrib['value'])
                elif "prep" == arg.attrib['value'].lower() and preps:
                    printlist.append(preps[0])
                else:
                    found = False
                    for role in self.roles.keys():
                        if role == arg.attrib['value']:
                            printlist.append(self.roles[role])
                            list_for_pred_change.append(self.roles[role])
                            found = True
                            break
                    if not found and arg.attrib['value'] != "prep":
                        printlist.append("?"+arg.attrib['value'])
                        list_for_pred_change.append("?"+arg.attrib['value'])
            temp_pred = predicate
            predicate+=",".join(printlist)+")"
            temp_pred+=",".join(list_for_pred_change)+")"
            #pred_obj = Predicate(temp_pred,negated)
            #if pred_obj:
            #    pred_objs.append(pred_obj)
            if negated:
                predicate = "not("+predicate+")"
            for key in currentFacts.keys(): #add each predicate to the appropriate noun in the state
                if key in predicate:
                    currentFacts[key].add(predicate)
        #self.changePredicateObject = ChangePredicate(pred_objs,self.roles,currentFacts,self.sels)
        return currentFacts

    """
    def simplifyFrame(self, parsed_sentence, POS):
        if "PP" in POS:
            #if the sentence starts with a PP or some clause, remove it
            if POS[0] == "PP":
                del POS[0]
                del parsed_sentence[0]
            else: #remove last PP and try again
                last = len(POS) - POS[::-1].index("PP") - 1
                del POS[last]
                del parsed_sentence[last]
        while POS.count("V") > 1: #remove multiple verbs
            last = len(POS) - POS[::-1].index("V") - 1
            POS=POS[:last]
            parsed_sentence=parsed_sentence[:last]
        #print("Simplified frame: "+str(POS))
        return parsed_sentence, POS
    """

##########Event Comprehension############
    """
    #the frame has been selected and is now slotted to be filled
    def addToState(self, frame,preps,newdict):
        #TODO: if the set is empty for an entity, remove the entity
        currentlyMentioned, state = self.fillPredicates(frame, preps, newdict)
        self.currentlyMentioned = copy.deepcopy(currentlyMentioned)
        #To be replaced with changePredicates:
        #newdict = self.addDictionaries(copy.deepcopy(self.currentlyMentioned), newdict) #put currently mentioned in newdict
        
        #found = True
        #self.facts = self.mergeFacts(self.facts,newdict) #remove facts that make things inaccessible
    """

    #check the verb's frame against the event and match up the semantic roles
    def searchFrame(self, frame, numRolesToFill, event, prep):
        newdict = defaultdict(set)
        preps = []
        wrongPrep = False
        original_POS_string, POS_frame = self.getPrimaryFrame(frame)
        foundVerb = False

        #if the sentence matches this frame
        if len(POS_frame) == numRolesToFill:
            #print("\nMatching frame " + str(original_POS_string) + " in ID " + classID)
            x = 0
            for POSs in frame.findall('SYNTAX'): #go through SYNTAX section in the XML
                if wrongPrep: break
                for pos in POSs: #for each part of speech in the frame
                    if wrongPrep: break
                    if pos.attrib: #not a VERB or ADV, given a value
                        #for each part of the event
                        for i in range(x, len(event)-1): #minus 1 to exclude prep
                            #if it's a match, add one to x
                            #break 
                            param = event[i]
                            #print(param)
                            param_pos = ""
                            if "Synset(" in param or "<" in param:
                                param_pos = "NP"
                            if pos.tag == param_pos: #matching POS
                                #add a new key to the dictionaries
                                if not param in newdict:
                                    newdict[param] = set()
                                self.roles[pos.attrib['value']]=param
                                #print(pos.attrib['value'])
                                """
                                #learn selectors from user
                                total_sel = self.maxSel(self.sels[pos.attrib['value']])
                                newdict[param]|=total_sel
                                self.currentlyMentioned[param]|=copy.deepcopy(total_sel)
                                """
                                newdict[param] |=self.sels[pos.attrib['value']]

                                x+=1
                                break
                            elif pos.tag == "PREP":
                                if pos.attrib['value'] == "":
                                    preps += [prep]
                                    break
                                
                                elif pos.attrib['value'][0].islower(): 
                                #if it's an actual list of prepositions                    
                                    if prep in pos.attrib['value']: #if the preposition is in this space-delimited list
                                        preps += [prep]
                                    else:
                                        wrongPrep = True
                                else:
                                    preps += [prep]
                                break
                            elif "-" in param:
                                if foundVerb:
                                    param = "EmptyParameter"
                                    if not pos.attrib['value'] in self.roles:                                        
                                        self.roles[pos.attrib['value']]= "?"+pos.attrib['value']
                                    
                                else:
                                    param_pos = "V"
                                    foundVerb = True
                                x+=1
                            elif param == "EmptyParameter":
                                    if not pos.attrib['value'] in self.roles:
                                        self.roles[pos.attrib['value']]="?"+pos.attrib['value']
                                    x+=1
                            else: #it's not a word that's found in WordNet
                                #add a new key to the dictionaries
                                if not param in newdict:
                                    newdict[param] = set()
                                self.roles[pos.attrib['value']]=param
                                #print(pos.attrib['value'])
                                
                                """
                                #learn selectors from user
                                total_sel = self.maxSel(self.sels[pos.attrib['value']])
                                newdict[param]|=total_sel 
                                self.currentlyMentioned[param]|=copy.deepcopy(total_sel)
                                """
                                newdict[param] |=self.sels[pos.attrib['value']]

                                x+=1
                                break
                                
                    elif pos.tag == "PREP": #no attributes
                        preps += [prep]
                    elif pos.tag == "VERB": #a verb
                        x+=1

            if wrongPrep:
                #print("Wrong preposition. Trying another frame.")
                return False, preps,newdict
            else:
                #self.addToState(frame,preps,newdict)

                #the frame has been selected and is now slotted to be filled
                #TODO: if the set is empty for an entity, remove the entity
                #currentlyMentioned = self.fillPredicates(frame, preps, newdict)
                self.currentlyMentioned = copy.deepcopy(self.currentlyMentioned)

                return True, preps,newdict
        return False,preps,newdict



    def getFramePOS(self, event):
        #print(event)
        [agent, verb, patient, theme, prep] = event
        if "-" not in verb: #it's not a VerbNet category
            #print('Cannot find verb "'+verb+'".')
            return None
        if "-" in patient and "Synset" not in patient: #it's a verb
            alpha, num = patient.split("-", 1)
            if alpha[-1].isalpha() and num[0].isdigit():
                verb = patient
                patient = "EmptyParameter"
        numRolesToFill = len(event) - event.count("EmptyParameter")
        if prep != "EmptyParameter":
            numRolesToFill-=1
        #print(numRolesToFill)
        
        classID = verb
        try:        
            vnclass = verbnet.vnclass(classID) #call up the verbnet file for this class
        except ValueError:
            #print('Cannot find verb "'+verb+'".')
            return []

        #pick out selectional restrictions
        sels = self.getSelectors(vnclass)
        superclass = None
        if classID.count("-") > 1: #add superclass selectors too
            classSplit = classID.split("-")
            superclass = verbnet.vnclass("-".join([classSplit[0],classSplit[1]]))
            sels.update(self.getSelectors(superclass))
        self.sels = copy.deepcopy(sels)
        
        found = False
        newdict,preps = None,None
        #find the appropriate frame
        for frame in vnclass.findall('FRAMES/FRAME'):
            found,preps,newdict = self.searchFrame(frame, numRolesToFill, event, prep)
            if found:
                _, POS_frame = self.getPrimaryFrame(frame)
                return POS_frame


        for c in vnclass.findall('SUBCLASSES/VNSUBCLASS'):
            for frame in c.findall('FRAMES/FRAME'):                
                found,preps,newdict = self.searchFrame(frame, numRolesToFill, event, prep)
                if found:
                    _, POS_frame = self.getPrimaryFrame(frame)
                    return POS_frame

        if superclass:
            for frame in superclass.findall('FRAMES/FRAME'):
                found,preps,newdict = self.searchFrame(frame, numRolesToFill, event, prep)
                if found:
                    _, POS_frame = self.getPrimaryFrame(frame)
                    return POS_frame
        
        #give up condition
        return None
        # _, POS_frame = self.getPrimaryFrame(frame); return POS_frame
                
    def addState(self, event):
        #print(event)
        [agent, verb, patient, theme, prep] = event
        if "-" not in verb: #it's not a VerbNet category
            #print('Cannot find verb "'+verb+'".')
            return        
        if "-" in patient and "Synset" not in patient: #it's a verb
            alpha, num = patient.split("-", 1)
            if alpha[-1].isalpha() and num[0].isdigit():
                verb = patient
                patient = "EmptyParameter"
        numRolesToFill = len(event) - event.count("EmptyParameter")
        if prep != "EmptyParameter":
            numRolesToFill-=1
        #print(numRolesToFill)
        
        classID = verb
        try:        
            vnclass = verbnet.vnclass(classID) #call up the verbnet file for this class
        except ValueError:
            #print('Cannot find verb "'+verb+'".')
            return

        #pick out selectional restrictions
        sels = self.getSelectors(vnclass)
        superclass = None
        if classID.count("-") > 1: #add superclass selectors too
            classSplit = classID.split("-")
            superclass = verbnet.vnclass("-".join([classSplit[0],classSplit[1]]))
            sels.update(self.getSelectors(superclass))
        self.sels = copy.deepcopy(sels)
        
        found = False
        newdict,preps = None,None
        #find the appropriate frame
        for frame in vnclass.findall('FRAMES/FRAME'):
            found,preps,newdict = self.searchFrame(frame, numRolesToFill, event, prep)
            if found:
                return


        for c in vnclass.findall('SUBCLASSES/VNSUBCLASS'):
            for frame in c.findall('FRAMES/FRAME'):                
                found,preps,newdict = self.searchFrame(frame, numRolesToFill, event, prep)
                if found:
                    return 

        if superclass:
            for frame in superclass.findall('FRAMES/FRAME'):
                found,preps,newdict = self.searchFrame(frame, numRolesToFill, event, prep)
                if found:
                    return

        

        #print("Gave up")
        #all the frames were wrong; we'll just pick the last one
        _, POS_frame = self.getPrimaryFrame(frame); #print(f'---> LAST: frame POS {POS_frame}') 

        found,preps,newdict = self.searchFrame(frame, len(POS_frame), event, prep)
        #self.addToState(frame,preps,newdict)

        #the frame has been selected and is now slotted to be filled
        #TODO: if the set is empty for an entity, remove the entity
        # print('\n\n******found frame {} and preps {} and newdict {} \n**for event: {}'.format(POS_frame,preps,newdict,event));
        currentlyMentioned = self.fillPredicates(frame, preps, newdict)
        self.currentlyMentioned = copy.deepcopy(currentlyMentioned)

        return


