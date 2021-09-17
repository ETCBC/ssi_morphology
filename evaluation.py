# This module provides a method for calculating the distance
# between two morphological encodings. It exports two functions:
# mc_distance(code1, code2) and mc_load_morphemes(language).

# mc_load_morphemes() needs to be called before the first call to
# mc_distance() in order to initialise the appropriate language
# (currently Hebrew or Syriac).

# mc_distance(code1, code2) takes as arguments two strings, which
# are supposed to represent morphological encodings, and returns an
# integer in the range 0..262143 (which is 8**6-1) indicating how far
# apart both encodings are. If one of the two strings is known to be
# a correct encoding, the distance expresses how `bad' the other is.

# mc_load_morphemes(language) takes one string as argument, the name
# of the language of which the paradigmatic forms of the morphemes
# are to be loaded. They are retrieved from a file {Language}.json,
# which must reside in the current directory.

# Base of the number system in which the distance is counted
Base = 8


# The `badness' of the encoding is evaluated in six respects,
# which gives rise to a vector of six dimensions. Each dimension is
# considered a factor `Base' less serious than the previous.
# v[0] = number of parse errors in the encoding
# v[1] = edit distance of the surface form to the true surface form
# v[2] = number of ungrammatical morpheme type combinations
# v[3] = number of unparadigmatic morphemes
# v[4] = difference in number of analytical words with the true form
# v[5] = number of morphemes that differ from the true form
Dimensions = 6


# Global variables that hold the morphemes of the language

# Concatenative morphemes
CCM_Types = []
CCM = []
Prefixes = []
Suffixes = []

# Nonconcatenative morphemes
NCM_Types = []
NCM = []


def mc_load_morphemes(l):
   '''Load the morphemes of the language (l) from a file.'''
   from json import load

   with open(l + '.json') as f:
      d = load(f)

   # Concatenative morphemes
   global CCM_Types, CCM, Prefixes, Suffixes
   CCM_Types = d['CCM_Types']
   CCM = d['CCM']

   Prefixes = CCM_Types[:d['N_Prefixes']]
   Suffixes = CCM_Types[-d['N_Suffixes']:]

   # Nonconcatenative morphemes
   global NCM_Types, NCM
   NCM_Types = d['NCM_Types']
   NCM = d['NCM']
   for m in NCM_Types:
      NCM[m] = set(NCM[m])


# In the class Word, morphemes are stored as a pair (tuple) of a
# realised form [0] and a paradigmatic form [1] like ('T', 'T=').
class Word():
   def __init__(self, prefixes, lexeme, suffixes):
      self.morphemes = {}
      self.morphemes['pfm'] = prefixes[0]
      self.morphemes['pfx'] = prefixes[1]
      self.morphemes['vbs'] = prefixes[2]
      self.morphemes['lex'] = lexeme
      self.morphemes['vbe'] = suffixes[0]
      self.morphemes['nme'] = suffixes[1]
      self.morphemes['uvf'] = suffixes[2]
      self.morphemes['vpm'] = suffixes[3]
      self.morphemes['prs'] = suffixes[4]

   def surface(self):
      s = ''
      for m in CCM_Types:
         if self.morphemes[m]:
            s += self.morphemes[m][0]
      return s

# The morphological code is parsed using McLexer and McParser
from sly import Lexer, Parser

# The token CHR makes sure that any input can be tokenised and no
# errors are raised. It is not used by the parser.
class McLexer(Lexer):
   def error(self, t):
      assert(False)

   tokens = { PFM, PFX, VBS, LETTER, HOMOGRAPHY, VBE, NME, UVF,
              VPM, VOWEL_PATTERN, PRS, CHR }
   literals = { '-', '(', '&' }

   PFM = '!'
   PFX = '@'
   VBS = '\\]'
   LETTER = '[>BGDHWZXVJKLMNS<PYQRFCT]'
   HOMOGRAPHY = '=+'
   VBE = '\\['
   NME = '/'
   UVF = '~'
   VOWEL_PATTERN = '[a-z]+'
   VPM = ':'
   PRS = '\\+'
   CHR = '[^-(&]'

# The parser returns a list of analytic words as objects of the
# class Word.
class McParser(Parser):
   #debugfile = 'parser.out'

   def __init__(self):
      super().__init__()
      self.status = 0

   def error(self, p):
      #print('Syntax error:', self.symstack)
      # Just discard the token and tell the parser it's okay.
      self.errok()
      self.status += 1

   tokens = McLexer.tokens - {'CHR'}

   @_('words')
   def wordlist(self, p):
      return (p[0], self.status)

   @_('word')
   def words(self, p):
      return [p[0]]

   @_('words "-" word')
   def words(self, p):
      p[0].append(p[2])
      return p[0]

   @_('prefixes lexeme suffixes')
   def word(self, p):
      return Word(p[0], p[1], p[2])

   @_('preformative reflexive verbal_stem')
   def prefixes(self, p):
      return (p[0], p[1], p[2])

   @_('form')
   def lexeme(self, p):
      return p.form

   @_('verbal_ending nominal_ending univalent_final \
       vowel_pattern_mark pronominal_suffix')
   def suffixes(self, p):
      return (p[0], p[1], p[2], p[3], p[4])

   @_('empty')
   def preformative(self, p):
      return None

   @_('PFM form PFM')
   def preformative(self, p):
      return p[1]

   @_('empty')
   def reflexive(self, p):
      return None

   @_('PFX form PFX')
   def reflexive(self, p):
      return p[1]

   @_('empty')
   def verbal_stem(self, p):
      return None

   @_('VBS form VBS')
   def verbal_stem(self, p):
      return p[1]

   @_('VBE form')
   def verbal_ending(self, p):
      return p[1]

   @_('empty')
   def verbal_ending(self, p):
      return None

   @_('NME form')
   def nominal_ending(self, p):
      return p[1]

   @_('empty')
   def nominal_ending(self, p):
      return None

   @_('UVF form')
   def univalent_final(self, p):
      return p[1]

   @_('empty')
   def univalent_final(self, p):
      return None

   @_('VPM VOWEL_PATTERN')
   def vowel_pattern_mark(self, p):
      return set(p[1])

   @_('empty')
   def vowel_pattern_mark(self, p):
      return None

   @_('PRS form')
   def pronominal_suffix(self, p):
      return p[1]

   @_('empty')
   def pronominal_suffix(self, p):
      return None

   @_('empty')
   def form(self, p):
      return ('', '')

   @_('letters homography')
   def form(self, p):
      realised = ''.join([t[0] for t in p[0]])
      paradigmatic = ''.join([t[1] for t in p[0]]) + p[1]
      return (realised, paradigmatic)

   @_('letter')
   def letters(self, p):
      return [p[0]]

   @_('letters letter')
   def letters(self, p):
      p[0].append(p[1])
      return p[0]

   @_('empty', 'HOMOGRAPHY')
   def homography(self, p):
      return p[0]

   @_('')
   def empty(self, p):
      return ''

   @_('plain_letter', 'deleted_letter', 'added_letter')
   def letter(self, p):
      return p[0]

   @_('LETTER')
   def plain_letter(self, p):
      return (p[0], p[0])

   @_('"(" LETTER')
   def deleted_letter(self, p):
      return ('', p[1])

   @_('"&" LETTER')
   def added_letter(self, p):
      return (p[1], '')


def ungrammatical_combinations(wl):
   '''Count the number of ungrammatical morpheme type combinations in
      word list (wl).'''
   r = 0
   for w in wl:
      for m in Prefixes:
         if w.morphemes[m] and not w.morphemes['vbe']:
            r += 1
      if w.morphemes['pfm']:
         for v in {'M'}:
            if v == w.morphemes['pfm'][1] and not w.morphemes['nme']:
               r += 1
      if w.morphemes['vpm']:
         for v in {'a', 'c'}:
            if v in w.morphemes['vpm'] and not w.morphemes['nme']:
               r += 1
         for v in {'d', 'p', 'o', 'u'}:
            if v in w.morphemes['vpm'] and not w.morphemes['vbe']:
               r += 1
   return r


def unparadigmatic_morphemes(wl):
   '''Count the number of unparadigmatic morphemes in word list (wl).'''
   r = 0
   for w in wl:
      for m in Prefixes + Suffixes:
         if w.morphemes[m] and w.morphemes[m][1] not in CCM[m]:
            r += 1
      for m in NCM_Types:
         if w.morphemes[m] and not (w.morphemes[m] < NCM[m]):
            r += 1
   return r


# The evaluation of the word list returned by the parser is done
# in two steps. First the dimensions [0,2,3] are assigned, which
# can be calculated without comparison with the true form. This is
# performed by evaluate(). Then the other dimensions are assigned
# through comparison with the true form. This is done by compare().

from numpy import zeros

def evaluate(wl, e):
   '''Evaluate the three dimensions which can be calculated
      individually for word list (wl) with (e) syntax errors.'''
   v = zeros(Dimensions)
   v[0] = e
   v[2] = ungrammatical_combinations(wl)
   v[3] = unparadigmatic_morphemes(wl)
   return v


def morpheme_comparison(w1, w2):
   '''Return the number of morphemes that differ in their analysis
      between word w1 and w2.'''
   r = 0
   for m in CCM_Types:
      if w1.morphemes[m] != w2.morphemes[m]:
         r += 1
   for m in NCM_Types:
      if w1.morphemes[m] != w2.morphemes[m]:
         r += 1
   return r


def wlsurface(wl):
   '''Return the surface of a word list.'''
   s = ''
   for w in wl:
      s += w.surface()
   return s


from Levenshtein import distance

def compare(w1, w2, v1, v2):
   '''Return a vector with an error count in all dimensions for the
      difference in analysis between word w1 and w2. The vectors v1
      and v2 already contain the counts of the individual evaluation
      of the words.'''
   v = abs(v1 - v2)
   v[1] = distance(wlsurface(w1), wlsurface(w2))
   v[4] = abs(len(w1) - len(w2))
   if v[4] == 0:
      for i in range(len(w1)):
         v[5] += morpheme_comparison(w1[i], w2[i])
   return v


def badness(d):
   '''Map the difference vector onto a nonnegative integer.'''
   assert(not any(d < 0))
   r = 0
   for x in d:
      r = Base * r + min(x, Base - 1)
   return r


def mc_parse(s):
   '''Call the lexer and parser for the morphological encoding of
      string (s) and return the word list and error count.'''
   lex = McLexer()
   parser = McParser()
   r = parser.parse(lex.tokenize(s))
   if r:
      return r
   else:
      return ([], Base)


def mc_distance(s1, s2):
   '''Return a nonnegative number that expresses how badly the
      morphological encodings in strings s1 and s2 differ.'''
   if s1 == s2:
      return 0
   else:
      w1, e1 = mc_parse(s1)
      v1 = evaluate(w1, e1)
      #assert(not any(v1))
      w2, e2 = mc_parse(s2)
      v2 = evaluate(w2, e2)
      v = compare(w1, w2, v1, v2)
      #print(v1, v2, v)
      return badness(v)
