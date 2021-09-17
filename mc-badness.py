from evaluation import mc_distance, mc_load_morphemes

# Default values for the options
Language = 'Hebrew'

# The input file should contain lines like "C@LAX CLX==/ CLX[ wrong".
# All lines with "wrong" as the fourth field are selected and the
# strings of the second and third field are passed to mc_distance().

def process(path):
   with open(path) as f:
      for line in f:
         l = line.split()
         if len(l) >= 4 and l[3] == "wrong":
            print(l[0], l[1], l[2], l[3], mc_distance(l[1], l[2]))


from sys import argv, stderr

def Usage():
   stderr.write('usage: mc-badness [-l language] file...\n')
   exit(1)


from getopt import getopt, GetoptError

def main(argv):
   try:
      opts, args = getopt(argv, "l:", [])
   except GetoptError:
      Usage()
   for opt, arg in opts:
      if opt == '-l':
         global Language
         Language = arg
   mc_load_morphemes(Language)
   for f in args:
      process(f)

if __name__ == "__main__":
   main(argv[1:])
