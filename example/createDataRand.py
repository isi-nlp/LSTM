import sys
from operator import itemgetter

input_words = dict((line.strip(),1) for line in open(sys.argv[1]))
output_words = dict((line.strip(),1) for line in open(sys.argv[2]))

import random

for i,word in enumerate(input_words):
  input_words[word] = i

for i,word in enumerate(output_words):
  output_words[word] = i


g_input = lambda x: input_words[x]
g_output = lambda x: output_words[x]

#print words
#raw_input()

#reading file
lines = [line.strip() for line in open(sys.argv[3])]
for i in range(len(lines)):
  lines[i] = "<s> %s </s>"%lines[i]

sent_lens = []
for line in lines:
  sent_lens.append(len(line))

#list1, sorted_list = zip(*sorted(zip(sent_lens, lines), key=itemgetter(0)))

random.shuffle(lines)
sorted_list = lines

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

