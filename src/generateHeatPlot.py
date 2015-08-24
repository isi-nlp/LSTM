import prettyplotlib as ppl
import numpy as np
import string
import sys
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

#go over the training file and keep getting data

import sys
import fileinput
import argparse
parser = argparse.ArgumentParser(description='Generate visualization from Decoder trace')
if (len(sys.argv) < 2):
  parser.print_help()
  sys.exit()

parser.add_argument('--decoder_trace_file', metavar='file', dest='decoder_trace_file', help='Decoder trace file. Required argument')
parser.add_argument('--output_viz_file', metavar='file', dest='output_viz_file', help='Output viz file. It will be in .pdf format. Required argument')

args = parser.parse_args()

data = [line.strip() for line in open(args.decoder_trace_file)]
title = args.output_viz_file
pdf = PdfPages(title+'.pdf')

h_column_labels = []
c_column_labels = []

num_hidden = 0


#first make the data
input_symbol = []
h_t = []
c_t =[]

prev_h_t = []
prev_c_t = []
row_labels = []
diagram_counter = 0

for line in data:
  if line.startswith("input_symbol:"):
    row_labels.append(line.split()[1])
  if line.startswith("h_t:"):
    h_t.append(map(float,line.split()[1:]))
  if line.startswith("c_t:"):
    c_t.append(map(float,line.split()[1:]))

  if "NEW SENTENCE" in line:
    diagram_counter += 1
    np_h_t  = np.array(h_t)
    #np_h_t = np.max(np.abs(np_h_t),axis=0)
    np_c_t  = np.array(c_t)
    #np_c_t /= np.max(np.abs(np_c_t),axis=0)
    #print np_c_t
    #raw_input()
    num_hidden = np_h_t.shape[0]
    
    for i in range(num_hidden):
      h_column_labels.append('h'+str(i+1))
      c_column_labels.append('c'+str(i+1))

    #np_full = np.concatenate([np_h_t,np_c_t],axis=1)
    column_labels = h_column_labels+c_column_labels
    #diagram_name = prefix+".example"+str(diagram_counter)+'.png'
    #print 'Generated heat map for example',diagram_counter,'in',diagram_name
    fig, ax = ppl.subplots(1,2)
    fig.suptitle('Sequence pair number '+str(diagram_counter), fontsize=12)
    ax[0].set_title("Hidden State Values")
    sns.heatmap(h_t,ax=ax[0], 
               xticklabels=h_column_labels, 
               yticklabels=row_labels,
                annot=False)
    ax[0].set_yticklabels(row_labels,rotation=0)
    ax[1].set_title("Cell Values")
    sns.heatmap(c_t,ax=ax[1],
               xticklabels=c_column_labels, 
               yticklabels=row_labels,
              annot=False)
    ax[1].set_yticklabels(row_labels,rotation=0)

    #ax[1].yticks(rotation=0) 
    #fig.yticks(rotation=0) 
    fig.tight_layout()
    fig.savefig(pdf, format='pdf') 
    #fig.savefig(diagram_name)
    #clearing the labels and states for the new sentence'
    row_labels = []
    h_t = []
    c_t = []

pdf.close()

#pp.close()
  #raw_input();

