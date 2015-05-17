import sys
from operator import itemgetter

from optparse import OptionParser
#[...]
parser = OptionParser()
parser.add_option("-f", "--input_file", type="string", dest="input_file",
                  help="Training file to generate data from", metavar="FILE")
parser.add_option("-i", "--integerized_input_file", dest="integerized_input_file",
                  help="integerized input file", type="string", metavar="FILE")
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

input_words = dict((line.strip(),1) for line in open(options.input_words_file))
output_words = dict((line.strip(),1) for line in open(options.output_words_file))


for i,word in enumerate(input_words):
  input_words[word] = i

for i,word in enumerate(output_words):
  output_words[word] = i


g_input = lambda x: input_words[x]
g_output = lambda x: output_words[x]

#print words
#raw_input()

#reading file
lines = [line.strip() for line in open(options.input_file)]
  
for i in range(len(lines)):
  lines[i] = "%s </s>"%lines[i]

data = []
for line in lines:
  data.extend(line.split())
data.insert(0,'<s>')

#getting the number of tokens
num_tokens = 0
for line in lines:
  num_tokens += len(line.split())
print options
print 'seq length',options.sequence_length
print 'minibatch size',options.minibatch_size
num_minibatches = (num_tokens-2)/(options.minibatch_size*options.sequence_length) +1 
print 'num batches', num_minibatches


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

for i in range(num_minibatches-1):
  int_input_line = map(g_input,data[i*options.sequence_length:(i+1)*options.sequence_length])
  int_output_line = map(g_output,data[i*options.sequence_length+1:(i+1)*options.sequence_length+1])
  input_file.write("%s\n"%' '.join(map(str,int_input_line)))
  output_file.write("%s\n"%' '.join(map(str,int_output_line)))

  
int_input_line = map(g_input,data[(num_minibatches-1)*options.sequence_length:-1])
int_output_line = map(g_output,data[(num_minibatches-1)*options.sequence_length+1:])
input_file.write("%s\n"%' '.join(map(str,int_input_line)))
output_file.write("%s\n"%' '.join(map(str,int_output_line)))


