import sys
from operator import itemgetter
import random.shuffle()

input_words = dict((line.strip(),1) for line in open(sys.argv[1]))
output_words = dict((line.strip(),1) for line in open(sys.argv[2]))


for i,word in enumerate(input_words):
  input_words[word] = i

for i,word in enumerate(output_words):
  output_words[word] = i


g_input = lambda x: input_words[x] if x in input_words else input_words['<unk>']
g_output = lambda x: output_words[x] if x in output_words else output_words['<unk>']


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

#print sorted_list
int_lines = []
input_file = open(sys.argv[4],'w')
output_file = open(sys.argv[5],'w')

length_sorted_lines = []

#pick up 1000 lines at a in the shuffled list, sort them, and then print them to the output file
for i in range((len(lines)-1)/1000+1): #item in sorted_list:
  #print item
  getSortedLengthLines(lines[i*1000:(i+1)*1000],lengths[i*1000:(i+1)*1000],length_sorted_lines)
  for line in length_sorted_lines:
    int_input_line = map(g_input,item.split()[:-1])
    int_output_line = map(g_output,item.split()[1:])
    input_file.write("%s\n"%' '.join(map(str,int_input_line)))
    output_file.write("%s\n"%' '.join(map(str,int_output_line)))
  #print

int_input_file.close()
int_output_file.close()

#  for line in item:
#    print line

def getSortedLengtLines(lines,lengths,length_sorted_lines):

  list1, length_sorted_lines= zip(*sorted(zip(lengths, lines), key=itemgetter(0)))

