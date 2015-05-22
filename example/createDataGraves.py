import sys
from operator import itemgetter

from optparse import OptionParser
#[...]
parser = OptionParser()
parser.add_option("-f", "--input_file", type="string", dest="input_file",
                  help="Training file to generate data from", metavar="FILE")
parser.add_option("-i", "--integerized_input_file", dest="integerized_input_file",
                  help="integerized input file", type="string", metavar="FILE")
parser.add_option("-t", "--sentence_cont_file", dest="sentence_cont_file",
                  help="sentece continuation file", type="string", metavar="FILE")
parser.add_option("-o", "--integerized_output_file", dest="integerized_output_file",
                  help="integerized output file", metavar="FILE")
parser.add_option("-v", "--input_words", type="string", dest="input_words_file",
                  help="input words file", metavar="FILE")
parser.add_option("-g", "--output_words",type="string", dest="output_words_file",
                  help="output words file", metavar="FILE")
parser.add_option("-s", "--minibatch_size",type="int", dest="minibatch_size",
                  help="minibatch_size", metavar="FILE")
parser.add_option("-l", "--sequence_length",type="int", dest="sequence_length",
                  help="Sequence length", metavar="FILE")



(options, args) = parser.parse_args()

input_words = dict((line.strip(),i) for i,line in enumerate(open(options.input_words_file)))
output_words = dict((line.strip(),i) for i,line in enumerate(open(options.output_words_file)))

#for i,word in enumerate(input_words):
#  input_words[word] = i

#for i,word in enumerate(output_words):
#  output_words[word] = i


g_input = lambda x: input_words[x]
g_output = lambda x: output_words[x]

#print words
#raw_input()

#reading file
lines = [line.strip() for line in open(options.input_file)]
  
for i in range(len(lines)):
  lines[i] = "%s </s>"%lines[i]

data = []
data_seq = []
for line in lines:
  data.append(line.split())
  data_seq.extend(line.split())

#data_seq.insert(0,'<s>')

#getting the number of tokens
num_tokens = 0
for line in lines:
  num_tokens += len(line.split())

#num_minibatches = (num_tokens-2)/(options.minibatch_size*options.sequence_length) +1 
#print 'num batches', num_minibatches

#num_tokens = len(data_seq)
#print 'num tokens',num_tokens
#num_minibatches = (num_tokens-1)/options.minibatch_size +1
#print 'minibatch width is ',num_tokens/options.minibatch_size

minibatch_indexes = [0]*options.minibatch_size
current_start_index = 0
for i in range(options.minibatch_size-1):
  approx_minibatch_end = current_start_index + num_tokens/options.minibatch_size
  #if the current word is the beginning of sentence, we're good
  if data_seq[approx_minibatch_end] != '</s>':
    #traverse forward till you find a end of line
    current_index = approx_minibatch_end
    while(data_seq[current_index] != '</s>'):
      current_index += 1
      #print 'current index',current_index
    minibatch_indexes[i] = current_index
  else :
    minibatch_indexes[i] = approx_minibatch_end
  current_start_index = minibatch_indexes[i]+1
  print 'minibatch index ',i,' is ',  minibatch_indexes[i]
  print data_seq[minibatch_indexes[i]-2:minibatch_indexes[i]+2]
  print  data_seq[minibatch_indexes[i]]

    
#minibatch_indexes[options.minibatch_size-1] = 
minibatch_indexes[options.minibatch_size-1] = len(data_seq)-1

#adding start of sentence symbols for the beginning of the minibatch
data_seq.insert(0,'<s>')
minibatch_indexes[0] += 1;
for i in range(1,options.minibatch_size):
  print 'inserting <s>'
  data_seq.insert(minibatch_indexes[i-1]+1,'<s>')
  print data_seq[minibatch_indexes[i-1]]
  minibatch_indexes[i] += i+1
  print data_seq[minibatch_indexes[i]-2:minibatch_indexes[i]+2]  

#print options
print 'seq length',options.sequence_length
print 'minibatch size',options.minibatch_size  
num_tokens = len(data_seq)
print 'num_tokens',num_tokens

#raw_input()
'''
sent_lens = []
for line in lines:
  sent_lens.append(len(line))

list1, sorted_list = zip(*sorted(zip(sent_lens, lines), key=itemgetter(0)))

#print sorted_list
int_lines = []
input_file = open(sys.argv[4],'w')
output_file = open(sys.argv[5],'w')


for item in sorted_list:
  #print item
  int_input_line = map(g_input,item.split()[:-1])
  int_output_line = map(g_output,item.split()[1:])
  input_file.write("%s\n"%' '.join(map(str,int_input_line)))
  output_file.write("%s\n"%' '.join(map(str,int_output_line)))
  #print

#  for line in item:
#    print line
'''

#go over every line and create a minibatch
#for now just create a minibatch of size 1

input_file = open(options.integerized_input_file,'w')
output_file = open(options.integerized_output_file,'w')
sent_cont_file = open(options.sentence_cont_file,'w')
current_minibatch_indexes = []

current_minibatch_indexes.append(0)

for i in range(1,options.minibatch_size):
  current_minibatch_indexes.append(minibatch_indexes[i-1]+1)
  print data_seq[current_minibatch_indexes[i]]

minibatch_sizes = [0]*options.minibatch_size

for i in range(options.minibatch_size):
  minibatch_sizes[i] = minibatch_indexes[i]-current_minibatch_indexes[i]#+1
  print 'last index of',i,'is ',data_seq[minibatch_sizes[i]]

print minibatch_sizes
print sum(minibatch_sizes)


remaining_minibatches = options.minibatch_size
while (remaining_minibatches >0):
  for i in range(options.minibatch_size):
    input_seq = []
    output_seq  = []
    cont_seq  = []
    if minibatch_sizes[i] > 0:
      current_sequence_length = min(minibatch_sizes[i],options.sequence_length)
      input_seq.extend(map(g_input,data_seq[current_minibatch_indexes[i]:current_minibatch_indexes[i]+current_sequence_length]))
      #print data_seq[current_minibatch_indexes[i]+1:current_minibatch_indexes[i]+current_sequence_length+1]
      output_seq.extend(map(g_output,data_seq[current_minibatch_indexes[i]+1:current_minibatch_indexes[i]+current_sequence_length+1]))
      cont_seq.extend([1]*current_sequence_length)
      minibatch_sizes[i] -= current_sequence_length
      current_minibatch_indexes[i] += current_sequence_length
      if minibatch_sizes[i] <= 0:
        remaining_minibatches -= 1
      if current_sequence_length < options.sequence_length:
        for i in range(options.sequence_length-current_sequence_length):
          input_seq.append(0)
          output_seq.append(-1)
          cont_seq.append(0)
    else:
      input_seq.extend([0]*options.sequence_length)
      output_seq.extend([-1]*options.sequence_length)
      cont_seq.extend([0]*options.sequence_length)
    #now to print out the lines
    input_file.write("%s\n"%(' '.join(map(str,input_seq))))
    output_file.write("%s\n"%(' '.join(map(str,output_seq))))
    sent_cont_file.write("%s\n"%(' '.join(map(str,cont_seq))))
    #raw_input()


input_file.close()
output_file.close()
sent_cont_file.close()


'''
for i in range(num_minibatches-1):
  int_input_line = map(g_input,data[i*options.sequence_length:(i+1)*options.sequence_length])
  int_output_line = map(g_output,data[i*options.sequence_length+1:(i+1)*options.sequence_length+1])
  input_file.write("%s\n"%' '.join(map(str,int_input_line)))
  output_file.write("%s\n"%' '.join(map(str,int_output_line)))
'''
#going over all the sentences until you reach the end

'''
minibatch_indexes = []
for i in range(options.minibatch_size):
  index_list = [i,0]
  minibatch_indexes.append(index_list)

finished_minibatches = options.minibatch_size

while(finished_minibatches > 0):
  for i in range(options.minibatch_size):
    sent = []
    while (len(sent) <= options.sequence_length):
      sent.append(data[minibatch_indexes[i][0]][[minibatch_indexes[i][0]])
'''

'''  
int_input_line = map(g_input,data[(num_minibatches-1)*options.sequence_length:-1])
int_output_line = map(g_output,data[(num_minibatches-1)*options.sequence_length+1:])
input_file.write("%s\n"%' '.join(map(str,int_input_line)))
output_file.write("%s\n"%' '.join(map(str,int_output_line)))
'''

