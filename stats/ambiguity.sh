#!/bin/sh

# Shell script to print the ratio of ambiguous forms in the input

NAME=`basename $0 .sh`

usage() {
   echo "usage: $NAME t_in t_out" >&2
   exit 1
}

# No options yet
while getopts v c
do
   case $c in
      *)
         usage ;;
   esac
done
shift `expr $OPTIND - 1`

test $# -eq 2 || usage

# Stop when an error occurs.
set -e

# Create three temporary files:
# One for the input tokens, output tokens, and the ambiguous tokens.
T_IN=`mktemp`
T_OUT=`mktemp`
T_AMB=`mktemp`

# Clean up after yourself when the script finishes.
trap "rm $T_IN $T_OUT $T_AMB" 0

# T_IN contains the input tokens, one per line.
cut -f4 "$1" | tr ' ' '\n' > $T_IN

# T_OUT the output tokens, split at the underscore if necessary.
cut -f4 "$2" | sed 's/_/& /g' | tr ' ' '\n' > $T_OUT

# Count how many tokens we have.
N_IN=`wc -l < $T_IN`
N_OUT=`wc -l < $T_OUT`

# Stop if input and output do not match.
test $N_IN -eq $N_OUT

# T_AMB contains the ambiguous input tokens.
paste $T_IN $T_OUT | sort -u | cut -f1 | uniq -d > $T_AMB

# Count how many ambiguous tokens we have among the input tokens.
N_AMB=`sort $T_IN | join $T_AMB - | wc -l`

# Print the ratio of ambiguous tokens.
echo $N_AMB $N_IN | awk '{print "Ambiguity is", $1/$2}'
