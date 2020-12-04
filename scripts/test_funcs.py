import random
people = ['robert','ciccy','jack','peder','peter','ben']
reviwers = ['robert','ciccy','jack','peder','peter','ben']

random.shuffle(people)
random.shuffle(reviwers)
for x,x1 in zip(people,reviwers): 
	if x == x1: 
		random.shuffle(people)
# 	while x == x1: 
# 		random.shuffle(people)

# #print(people)
# pairings = dict(zip(people,reviwers))

# for k,v in pairings.items(): 
# 	while k == v: 
# 		random.shuffle(people)
# 		#random.shuffle(reviwers)
# 		pairings = dict(zip(people,reviwers))
# 	else: 
# 		print('No more matches')
		
#print(pairings)
