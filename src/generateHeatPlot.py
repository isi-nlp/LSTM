import prettyplotlib as ppl
import numpy as np
import string
import sys

'''
fig, ax = ppl.subplots(1)

np.random.seed(10)

ppl.pcolormesh(fig, ax, np.random.randn(10,10), 
               xticklabels=string.uppercase[:10], 
               yticklabels=string.lowercase[-10:])
fig.savefig('pcolormesh_prettyplotlib_labels.png')
'''
#go over the training file and keep getting data

data = [line.strip() for line in open(sys.argv[1])]
num_hidden = int(sys.argv[2])
column_labels = []
for i in range(num_hidden):
  column_labels.append('h'+str(i+1))

counter = 0
example_hidden_states = []
print 'len ',len(data)
diagram_counter = 0
while (counter < len(data)):
  print 'in outer while loop'
  while ("NEW SENTENCE" not in data[counter]):
    example_hidden_states.append(data[counter])
    counter += 1
  # now do processing
  row_labels = []
  hidden_states = []
  for line in example_hidden_states:
    #print 'line',line
    #print line.split()
    row_labels.append(line.split(' ')[0]) 
    hidden_states.append(map(float,line.split(' ')[1:]))
    np_hidden_states  = np.array(hidden_states)
  diagram_counter += 1
  diagram_name = 'test'+str(diagram_counter)+'.png'
  print np_hidden_states
  print row_labels
  print column_labels
  example_hidden_states = []
  counter += 1
  #print data[counter]
  #print 'counter is ',counter
  #print 'generated one diagram'
  
  fig, ax = ppl.subplots(1)
  ppl.pcolormesh(fig, ax,np_hidden_states, 
               xticklabels=column_labels, 
               yticklabels=row_labels)
  
  fig.savefig(diagram_name)
  raw_input()



#pp.close()
  #raw_input();

