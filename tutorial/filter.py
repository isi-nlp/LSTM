import sys

words = dict([line.strip(),1] for line in open(sys.argv[1]))

g = lambda x: x if x in words else '<unk>'
for line in open(sys.argv[2]):
  if line.strip() == "":
    continue
  print ' '.join(map(g,line.strip().split()))

