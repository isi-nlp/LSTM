#!/usr/bin/env python

import vocab
import collections

start = "<s>"
stop = "</s>"
null = "<null>"

def ngrams(words, n):
    for i in xrange(n-1, len(words)):
        yield words[i-n+1:i+1]


def writeWords(words_file,v):
  with open(words_file, "w") as outfile:
    for w in v.words:
      outfile.write("%s\n" % (w,))

def insertInVocab(v,c,vocab_size,vocab_type):
    #v.insert_word(null)
    inserted = v.from_counts(c, vocab_size)
    if inserted == len(c):
        sys.stderr.write("warning: only %d words types in training data; set --n_%s_vocab lower to learn unknown word\n"%(len(c),vocab_type));

def writeData(data,filename,v):
    outfile = open(filename, 'w')
    for words in data:
        #for ngram in ngrams(words, n):
        outfile.write(" ".join(str(v.lookup_word(w)) for w in words) + "\n")


if __name__ == "__main__":
    import sys
    import fileinput
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess training data for n-gram language model.')
    if (len(sys.argv) < 2):
      parser.print_help()
      sys.exit()

    parser.add_argument('--input_train_text', metavar='file', dest='input_train_text', help='input training text file')
    parser.add_argument('--output_train_text', metavar='file', dest='output_train_text', help='output training text file')
    parser.add_argument('--input_validation_text', metavar='file', dest='input_validation_text', help='validation input text file (overrides --validation_size)')
    parser.add_argument('--output_validation_text', metavar='file', dest='output_validation_text', help='output validtion text file (overrides --validation_size)')
    parser.add_argument('--n_input_vocab', metavar='V', dest='n_input_vocab', type=int, help='number of input word types')
    parser.add_argument('--n_output_vocab', metavar='V', dest='n_output_vocab', type=int, help='number of output word types')
    parser.add_argument('--input_words_file', metavar='file', dest='input_words_file', help='make input vocabulary')
    parser.add_argument('--output_words_file', metavar='file', dest='output_words_file', help='make output vocabulary')
    parser.add_argument('--input_train_file', metavar='file', dest='input_train_file', default='-', help='make input training file')
    parser.add_argument('--output_train_file', metavar='file', dest='output_train_file', default='-', help='make input training file')
    parser.add_argument('--input_validation_file', metavar='file', dest='input_validation_file', help='make input validation file')
    parser.add_argument('--output_validation_file', metavar='file', dest='output_validation_file', help='make output validation file')
    parser.add_argument('--validation_size', metavar='m', dest='validation_size', type=int, default=0, help="select m lines for validation. Selects the last m lines")
    args = parser.parse_args()

    input_train_data = []
    output_train_data = []
    input_validation_data = []
    output_validation_data = []
    
    #getting the data
    for li, line in enumerate(file(args.input_train_text)):
        words = line.split()
        #words = [start] + words
        input_train_data.append(words)

    for li, line in enumerate(file(args.output_train_text)):
        words = line.split()
        words = [start] + words + [stop]
        output_train_data.append(words)

    if args.input_validation_text:
        for li, line in enumerate(file(args.input_validation_text)):
            words = line.split()
            #words = [start] * (n-1) + words + [stop]
            input_validation_data.append(words)
    else:
        if args.validation_size > 0:
            input_validation_data = input_train_data[-args.validation_size:]
            input_train_data[-args.validation_size:] = []
    
    if args.output_validation_text:
        for li, line in enumerate(file(args.output_validation_text)):
            words = line.split()
            #words = [start] * (n-1) + words + [stop]
            words = [start] + words + [stop]
            output_validation_data.append(words)
    else:
        if args.validation_size > 0:
            output_validation_data = output_train_data[-args.validation_size:]
            output_train_data[-args.validation_size:] = []

    #getting the vocabulary

    input_c = collections.Counter()
    for words in input_train_data:
        input_c.update(words[:])

    output_c = collections.Counter()
    for words in output_train_data:
        output_c.update(words[:])

    lengths = [len(input_words) + len(output_words) for input_words,output_words in zip(input_train_data,output_train_data)]

    input_v = vocab.Vocab()
    #input_v.insert_word(start)
    #input_v.insert_word(stop)
    #input_v.insert_word(null)
    insertInVocab(input_v,input_c,args.n_input_vocab,"input")
    #input_inserted = v.from_counts(c, args.n_vocab)

    output_v = vocab.Vocab()
    output_v.insert_word(start)
    output_v.insert_word(stop)
    insertInVocab(output_v,output_c,args.n_output_vocab,"output")


    '''
    if args.train_file == '-':
        outfile = sys.stdout
    else:
        outfile = open(args.train_file, 'w')
    '''
    #writing the vocabularies
    writeWords(args.input_words_file,input_v)
    writeWords(args.output_words_file,output_v)

    #writing the training and validation data
   
    writeData(input_train_data,args.input_train_file,input_v)
    writeData(input_validation_data,args.input_validation_file,input_v)

    writeData(output_train_data,args.output_train_file,output_v)
    writeData(output_validation_data,args.output_validation_file,output_v)
    
    '''
    for words in input_train_data:
        #for ngram in ngrams(words, n):
        outfile.write(" ".join(str(v.lookup_word(w)) for w in ngram) + "\n")
    if outfile is not sys.stdout:
        outfile.close()

    if args.validation_file:
        with open(args.validation_file, 'w') as outfile:
            for words in validation_data:
                for ngram in ngrams(words, n):
                    outfile.write(" ".join(str(v.lookup_word(w)) for w in ngram) + "\n")
    '''
