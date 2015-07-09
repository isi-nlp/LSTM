#!/usr/bin/env python



if __name__ == "__main__":
    import sys
    import fileinput
    import argparse

    parser = argparse.ArgumentParser(description='Generate text from integerized files.')
    parser.add_argument('--words_file', metavar='file', dest='words_file', help='words file')
    parser.add_argument('--integerized_text', metavar='file', dest='integerized_text', help='Integerized text file')
    parser.add_argument('--words_text', metavar='file', dest='words_text', help='Output words text file') 
    args = parser.parse_args()
    vocab = dict((i,line.strip()) for i,line in enumerate(open(args.words_file)))
    outfile = open(args.words_text,'w')
    for line in open(args.integerized_text):
      outfile.write("%s\n"%' '.join([vocab[int(int_word)] for int_word in line.strip().split()]))

