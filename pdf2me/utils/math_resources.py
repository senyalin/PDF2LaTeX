#################
# common function extraction
#################
import re

#import pytrie as trie
str2gn = {
    '\x86': "\\dagger",
    '\x87': "\\ddagger",
    '\x88': "\\hat",
    '\x91': '`',
    '\x92': "\\quoteright",
    '\x93': "SetTransmitState",  # left double quote open
    '\x94': "CancleCharacter",  # right quote close
    '\x95': "\\bullet",
    '\x96': "BeginGuard",  # http://www.fileformat.info/info/unicode/char/96/index.htm
    '\x97': "EndGuard",  # http://www.fileformat.info/info/unicode/char/96/index.htm
    '\x98': "\\tilde",
    '\xa3': "\\textsterling",  # UK pound https://tex.stackexchange.com/questions/17853/pounds-and-textsterling-produces-a-dollar-sign
    '\xa7': '\\textsection',
    '\xa8': "\\ddot",
    '\xae': "RegisteredSign",
    '\xaf': "\\bar",  # macron
    '\xb0': "degree",
    '\xb1': "\\pm",
    '\xba': "MasculineOrdinalIndicator",
    '\xb4': "\\acute",
    '\xb5': "\\mu",
    '\xb6': "\\mathparagraph", # http://tug.ctan.org/info/symbols/comprehensive/symbols-a4.pdf
    '\xb7': "\\cdot",
    '\xb8': "\\c",  # cedilla, http://www.fileformat.info/info/unicode/char/b8/index.htm
    '\xd7': "\\times",
    '\xd8': "\\emptyset",
    '\xdf': "\\ss",
    '\xe6': "\\ae",
    '\xf8': "\\emptyset",
    '\xfa': "uAcute",
}

unicode2gn = {

    u'\xb0': "degree",
    #u'\xba': "MasculineOrdinalIndicator",
    u'\xba': "degree",

    # TODO, part of the matrix
    u'\u239b': "(",  # http://www.fileformat.info/info/unicode/char/239b/index.htm, upper hook
    u'\u239c': "|",  # extension
    u'\u239d': "(",  # http://www.fileformat.info/info/unicode/char/239d/index.htm, lower hook
    u'\u239e': ")",  # upper
    u'\u239f': "|",  # extension
    u'\u23a0': ")",  # under
    u'\u23a1': "[",  # http://www.fileformat.info/info/unicode/char/23a1/index.htm, left square bracket, upper
    u'\u23a2': "|",  # create big bracket, left bracket extention
    u'\u23a3': "[",  # under part
    u'\u23a4': "]",  # right up
    u'\u23a5': "|",  # right extension
    u'\u23a6': "]",  # right lower

    u'\xfa': "uAcute",
}

unicode2latex = {

    u'\xae': "\\copyright",
    u'\xaf': "\\bar", # overline

    u'\xb1': "\\pm",
    u'\xd7': "\\times",  # times
    u'\xd8': "\\empty",  # times
    u'\xb5': "\\mu",  # mu
    u'\xb7': "\\cdot",  # cdot

    u'\xdf': "\\Beta",
    u'\x92': "'",  # single quote
    #'\x92': "'",  # single quote
    u'\u0142': "\\nmid",

    #https://www.fileformat.info/info/unicode/version/1.1/index.htm

    u'\u02da': '\\dot',  # empty circle dot
    u'\u02b9': '\\prime',
    u'\u02ba': '\'\'',  # double prime

    u'\u0302': '\\hat',
    u'\u0303': '\\tilde',

    u'\u0391': "\\Alpha",
    u'\u0392': "\\Beta",
    u'\u0393': "\\Gamma",
    u'\u0394': "\\Delta",
    u'\u0395': "\\Epsilon",
    u'\u0396': "\\Zeta",
    u'\u0397': "\\Eta",
    u'\u0398': "\\Theta",
    u'\u0399': "\\Iota",
    u'\u039A': "\\Kappa",
    u'\u039B': "\\Lambda",
    u'\u039C': "\\Mu",
    u'\u039D': "\\Nu",
    u'\u039E': "\\Xi",
    u'\u039F': "\\Omicron",

    u'\u03a0': "\\Pi",
    u'\u03a1': "\\Rho",
    u'\u03a3': "\\Sigma",
    u'\u03a4': "\\Tau",
    u'\u03a5': "\\Upsilon",
    u'\u03a6': "\\Phi",
    u'\u03a7': "\\Chi",
    u'\u03a8': "\\Psi",
    u'\u03a9': "\\Omega",

    u'\u03b1': "\\alpha",  # alpha
    u'\u03b2': "\\beta",  # beta
    u'\u03b3': "\\gamma",  # gamma
    u'\u03b4': "\\delta",
    u'\u03b5': "\\epsilon",
    u'\u03b6': "\\zeta",
    u'\u03b7': "\\eta",
    u'\u03b8': "\\theta",
    u'\u03b9': "\\iota",
    u'\u03ba': "\\kappa",
    u'\u03bb': "\\lambda",  # lamba
    u'\u03bc': "\\mu",  # nu
    u'\u03bd': "\\nu",  # nu
    u'\u03be': "\\xi",  # nu
    u'\u03bf': "\\omicorn",  # nu
    u'\u03c0': "\\pi",  # pi
    u'\u03c1': "\\rho",  # pi
    u'\u03c2': "\\sigma",
    u'\u03c3': "\\sigma",
    u'\u03c4': "\\tau",
    u'\u03c5': "\\upsilon",
    u'\u03c6': "\\phi",
    u'\u03c7': "\\chi",
    u'\u03c8': "\\psi",
    u'\u03c9': "\\omega",

    u'\u03d0': "\\Beta",
    u'\u03d1': "\\Theta",
    u'\u03d2': "\\Upsilon", #hook
    u'\u03d3': "\\Upsilon", #acute
    u'\u03d5': "\\Phi",
    u'\u03d6': "\\Pi",
    u'\u03da': "\\Stigma",
    u'\u03dc': "\\Digamma",
    u'\u03de': "\\Koppa",

    u'\u03f5': '\\epsilon',

    # math related after

    u'\u2013': "-",
    u'\u2014': "-",
    u'\u2015': "-",  # http://www.fileformat.info/info/unicode/char/2015/index.htm
    u'\u201c': "\"",
    u'\u201d': "\"",
    u"\u2020": "\\dagger",
    u'\u2021': '\\ddagger', # in paper 7098
    u'\u2022': '\\bullet',
    u'\u2026': "\\ldots",
    u"\u2032": "\\prime",
    u'\u204e': "\\ast", # lower asterisk, http://www.fileformat.info/info/unicode/char/204e/index.htm

    u'\u2112': "L",  # script l
    u'\u2113': "l",  # script l
    u'\u2126': "\\Omega",  # Omega
    u'\u2190': "\\leftarrow",
    u'\u2191': "\\uparrow",
    u'\u2192': "\\rightarrow",
    u'\u2193': "\\downarrow",
    u'\u2194': "\\leftrightarrow",
    u'\u2195': "\\updownarrow",

    u'\u21b5': "\\downarrow",
    u'\u21d2': "\\Rightarrow",
    u'\u21d4': "\\Leftrightarrow",
    u'\u21e0': "\\leftarrow",
    u'\u21e1': "\\uparrow",
    u'\u21e2': "\\rightarrow",
    u'\u21e3': "\\downarrow",
    u'\u21e4': "\\leftarrow",

    u'\u21e5': "\\rightarrow",
    u'\u21e6': "\\leftarrow",
    u'\u21e7': "\\uparrow",
    u'\u21e8': "\\rightarrow",
    u'\u21e9': "\\downarrow",
    u'\u21ea': "\\uparrow",

    u'\u2200': "\\forall",
    u'\u2201': "\\setminus",
    u'\u2202': "\\partial",
    u'\u2203': "\\exists",  # exist
    u'\u2204': "!\\exists",  # not exist
    u'\u2205': "\\empty",
    u'\u2206': "\\Delta",  # incremental, https://www.fileformat.info/info/unicode/char/2206/index.htm
    u'\u2207': "\\bigtriangledown",  # "\\in",
    u'\u2208': "\\in",  # "\\in",
    u'\u2209': "\\ni",  # "\\in",
    u'\u220f': '\\prod',
    u'\u2211': '\\sum',
    u'\u2212': "-",  # minus
    u'\u2213': '\\mp',
    u'\u2215': '/',
    u"\u2217": "\\ast",
    #u"\u2219": "\\dot", # for accent
    u"\u2219": "\\bullet",
    u'\u221d': "\\propto",  # "\propto",
    u"\u221e": "\\infty",

    u'\u2223': '\\mid',
    u'\u2224': '\\nmid',
    u'\u2225': '\\parallel',
    u'\u2226': '\\not\\parallel',
    u'\u2227': "\\wedge",

    u'\u2229': '\\cap',
    u'\u222a': '\\cup',
    u'\u222b': '\\int',
    u'\u222c': '\\iint',
    u'\u222d': '\\iiint',

    u'\u223c': "\\simeq",
    u'\u2243': "\\simeq", # 6110
    u'\u2248': "\\simeq", # 6110
    u'\u2260': "\\ne",
    u'\u2261': "\\equiv",  #\equiv
    u'\u2264': "\\leq",  # less equal
    u'\u2265': "\\geq",  # greater equal

    u'\u2282': '\\subset',
    u'\u2283': '\\supset',
    u'\u2284': '\\not\\subset',
    u'\u2285': '\\not\\supset',
    u'\u2286': "\\subseteq",  # subseteq

    u'\u2295': "\\oplus",
    u'\u2296': "\\ominus",
    u'\u2297': "\\otimes",
    u'\u2299': "\\odot",
    u'\u22a5': "\\perp",
    u'\u22c5': "\\cdot",  # dot operator
    u'\u22ef': "\\cdots",

    u'\u2303': "\\uparrow",
    u'\u2304': "\\downarrow",

    u'\u02c6': "\\hat",
    u'\u02d8': "\\breve",
    u"\u02d9": "\\dot",
    u'\u0305': '\\overline',
    u'\u23a7': "|",
    u'\u23aa': "|",
    u'\u23a8': "{",


    u'\u2606': '\\star',
    u'\u2713': "\\sqrt",
    u'\u2726': "\\star",
    u'\u221a': '\\sqrt',

}


import string
for c in string.printable:
    unicode2latex[unicode(c)] = c


# NOTE, the below variable is deprecated
# trying to unify it with the unicode2latex
special_unicode_chars = [
    # strange windows coding
    '\x92',  # https://stackoverflow.com/questions/29419322/unicodedecodeerror-utf8-codec-cant-decode-byte-0x92-in-position-377826-inva
    '\x96',  # dash, for the file 2021_5, MS font


    u'\x8a',  # http://www.codetable.net/hex/8a s rev hat
    u'\x8d',  # RI
    u'\x93',  # STS, might be control
    u'\x97',  # EPA, might be control

    u'\xa2',  # cross C
    u'\xa3',  # France dollar
    u'\xa7',  # dollar
    u'\xa8',  # like double prime. or yue yin
    u'\xa9',  # copyright C
    u'\xae',  # registered R
    u'\xaf',  # overline
    u'\xb0',  # just circle over
    u'\xb4',  # acute accent
    u'\xb6',  # looks like comma
    u'\xb8',  # looks like comma
    u'\xbc',  # 1/4
    u'\xbd',  # 1/2
    u'\xc1',  # A hat
    u'\xc5',  # A with circle over
    u'\xc7',  # C with "gou zi" under
    u'\xc9',  # E with "gou zi" under
    u'\xd3',  # O er sheng
    u'\xd6',  # O san sheng
    u'\xdc',  # U
    u'\xde',  # http://www.codetable.net/hex/de
    u'\xe0',  # a, 4 sheng
    u'\xe1',  # a, 2 sheng
    u'\xe3',  # a tilde
    u'\xe4',  # a, yu
    u'\xe6',  # ae
    u'\xed',  # iacute, european chars
    u'\xe7',  # c with "gou zi" under
    u'\xe8',  # e si sheng
    u'\xe9',  # e hat
    u'\xea',  # e hat
    u'\xeb',  # e double dot
    u'\xef',  # i two dots, too
    u'\xf0',  #http://www.codetable.net/hex/f0
    u'\xf1',  #e, n
    u'\xf3',  # i two dots
    u'\xf4',  # hat o
    u'\xf6',  # o two dot
    u'\xfa',  # u er sheng
    u'\xfc',  # u two dot
    u'\xf8',  # char like empty set, but in names
    u'\xfe',  # like p


    u'\u0107',  # some other A
    u'\u010d',  # a yue
    u'\u0117',  # a yue
    u'\u01eb',  # http://graphemica.com/%C7%AB
    u'\u0131',  # dotless i
    u'\u0141',  # L with stroke, http://www.fileformat.info/info/unicode/char/0141/index.htm
    u'\u0144',  # n er sheng
    u'\u0152',  # CE
    u'\u0153',  # ae
    u'\u015f',  # S with under
    u'\u0161',  # caron small s http://www.fileformat.info/info/unicode/char/0161/index.htm
    u'\u017e',  # z san sheng
    u'\u02c7',  # caron
    u"\u02db",  # under gouzi
    u'\u02dc',  # small tilde
    u'\u03cd',  # 2 sheng u

    u'\ufb01',  # fi
    'fi',
    u'\ufb02',  # fl
    'fl',
    u'\ufb03',  # ffi
    'ffi',
    u'\ufb00',  # ff
    'ff',

    u'\u2010',  # aC
    u'\u2014',  # long hyphen
    u'\u2022',  # bullet
    u'\u2018',  # right single quote
    u'\u2019',  # right single quote
    u'\u2013',  # en dash

    u'\u201c',  # left double quote
    u'\u201d',  # right double quote
    u'\u20ac',  # C
    u'\u226b',  # a % <<
    u'\u2305',  # projective ?
    u'\u23a9',  # looks like floor , but not
    u'\u2318',  # place of interest
    u'\u232b',  # cross arrow box
    u'\u2326',  # control erase
    u'\u2426',  # TODO
    u'\u25ba',  # solid right triangle
    u'\u25e6',  # bullet
    u'\u25c6',  # black diamond
    u'\u25cf',  # black bullet
    u'\u25ee',  # half filled triangle, or R?
    u'\u2660',  # black heart
    u'\u2663',  # flower
    u'\u2665',  # heart
    u'\u2666',  # filled diamond
    u'\u266b',  # music
    u'\u270f',  # pencil
    u'\u2afb',  # unknown

    u'\uf0b7',  # TODO
    u'\uf6be',  # 3/4 ?
    u'\uf8ea',
    u'\uf8eb',
    u'\uf8ec',
    u'\uf8ee',
    u'\uf8ed',
    u'\uf8ef',

    u'\uf8f0',
    u'\uf8f1',
    u'\uf8f2',
    u'\uf8f3',
    u'\uf8f4',
    u'\uf8f5',
    u'\uf8f6',
    u'\uf8f7',
    u'\uf8f8',
    u'\uf8f9',
    u'\uf8fa',
    u'\uf8fb',
    u'\uf8fc',
    u'\uf8fd',
    u'\uf8fe',
    u'\uf8ff',
    u'\ufb04',  # ffl
    u'\ufffc',
    u'\ufffd',
    u'\ufffe',
    u'\uf8ff',

]


triangle_func_list = [
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',  # trigonometric function
    'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',  # hyperbolic function
    'arccos', 'arccosh', 'arccot', 'arccoth', 'arccsc',
    'arccsch', 'arcsec', 'arcsech', 'arcsin', 'arcsinh', 'arctan', 'arctanh',
]

# NOTE
# this list is original designed for natural language processing to match math word
# but later also used in the parsing to match out math function.
func_name_list = [
    # logic related
    'nand', 'xor', 'xnor', 'nor',

    # arithemetic related
    'exp', 'ln', 'log',
    'det', 'tr', 'diag', 'rank', # algebra
    'round', 'gcd', 'lcm', 'mod', # number theory
    'remainder', 'quotient',

    # functional
    'domain', 'Id', "range", 'image',
    'domainofapplication', 'left_compose', 'left_inverse', 'right_inverse',
    'apply_to_list', 'kernel', 'right_compose',
    "Laplacian", "curl",

    # set
    'size', 'make_list',

    # number
    'ceil', 'floor', 'round', 'trunc',

    # stat
    'mean', 'median', 'mode',
    'std', 'sdev', 'variance', 'moment',

    'argmin', 'argmax',
    'min', 'max', 'lim',
    'arg', 'deg', 'dim',
    'Harr', "Jac", 'sgn',
    'sigmoid',

    # full names
    "cosine",
    "sine",
    "secant",
    "ker",
    "kernel",
    "Kernel",
    'tangent',
    "inverse-sine",
    "maximum", "minimum",
    "exponential",
    "hyperbolic-tangent",
    "absolute-value",

    # complex number
    "real", "real-part", "imaginary",

    "logarithm",
    "Pr",  # probability
    "degree",
    "argument",
    
    # adhoc name here
    "len", "WordError", # arxiv 1712.01769
    "sup",
    'IFFT', 'FFT',
    'ifft', 'fft',
    'otherwise', 'w.r.t', 's.t.', 'i.e.', 'e.g.',
]
func_name_list.extend(triangle_func_list)


def clean_function_name(func_name):
    clean_name_list = [
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',  # trigonometric function
        'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',  # hyperbolic function
        'exp', 'ln', 'log',
        'det', 'tr', 'diag', 'rank',
        'round', 'gcd', 'mod',
        'mean', 'std',
        'min', 'max', 'lim',
        'arg', 'deg', 'dim',
        'Harr', "Jac", 'sgn',
        "ker", "len", "sup",
    ]

    complex_name2simple_name = {
                                   'sigmoid': "\\sigma",
                                   "cosine": "cos",
                                   "sine":"sin",
                                   "secant": "sec",
                                   'tangent': "tan",
                                   "kernel": "Ker",
                                   "Kernel": "Ker",
                                   "inverse-sine": "\\sin^{-1}",
                                   "maximum": "max",
                                   "minimum": "min",
                                   "exponential": "exp",
                                   "hyperbolic-tangent": "tanh",
                                   "absolute-value": "abs",
                                   "logarithm": "log",
                                   "Pr": "P",  # probability
                                   "degree": "deg",
                                   "argument": "arg",
    }

    if func_name in clean_name_list:
        return func_name
    if func_name in complex_name2simple_name:
        return complex_name2simple_name[func_name]

    raise Exception("unknown function name {}".format(func_name))


words_list = [
    'measure'
]

max_func_name_len = 0
for func_name in func_name_list:
    max_func_name_len = max(max_func_name_len, len(func_name))
for word in words_list:
    max_func_name_len = max(max_func_name_len, len(word))

"""
func_name_trie = trie.CharTrie()
for func_name in func_name_list:
    func_name_trie[func_name] = func_name

#http://pygtrie.readthedocs.io/en/latest/
print func_name_trie
print func_name_trie.has_node('sin')
print func_name_trie.longest_prefix('sinalpha')
print list(func_name_trie.prefixes('sinhalpha'))
print list(func_name_trie.iteritems())

test_str = "sinhalpha"
for i in range(1, len(test_str)):
    print i, test_str[:i], \
        func_name_trie.has_subtrie(test_str[:i]), \
        func_name_trie.longest_prefix(test_str[:i])

import pygtrie
t = pygtrie.StringTrie()
t['foo'] = 'Foo'
t['foo/bar/baz'] = 'Baz'
print t.longest_prefix('foo/bar/baz/qux')
print t.longest_prefix('fooexist')
#"""

# The list of latex command is from the following website
# http://web.ift.uib.no/Teori/KURS/WRK/TeX/symALL.html
latex_math_symbols = """
\\alpha               \\theta               o                  \\tau
 \\beta                \\vartheta           \\pi                 \\upsilon
 \\gamma               \\gamma              \\varpi              \\phi
 \\delta               \\kappa              \\rho                \\varphi
 \\epsilon             \\lambda             \\varrho             \\chi
 \\varepsilon          \\mu                 \\sigma              \\psi
 \\zeta                \\nu                 \\varsigma           \\omega
 \\eta                 \\xi

 \\Gamma               \\Lambda             \\Sigma              \\Psi
 \\Delta               \\Xi                 \\Upsilon            \\Omega
 \\Theta               \\Pi                 \\Phi

  \\pm                  \\cap                \\diamond                    \\oplus
 \\mp                  \\cup                \\bigtriangleup              \\ominus
 \\times               \\uplus              \\bigtriangledown            \\otimes
 \\div                 \\sqcap              \\triangleleft               \\oslash
 \\ast                 \\sqcup              \\triangleright              \\odot
 \\star                \\vee                \\lhd$^b$                    \\bigcirc
 \\circ                \\wedge              \\rhd$^b$                    \\dagger
 \\bullet              \\setminus           \\unlhd$^b$                  \\ddagger
 \\cdot                \\wr                 \\unrhd$^b$                  \\amalg
 +                    -

 \\leq                 \\geq                \\equiv              \\models
 \\prec                \\succ               \\sim                \\perp
 \\preceq              \\succeq             \\simeq              \\mid
 \\ll                  \\gg                 \\asymp              \\parallel
 \\subset              \\supset             \\approx             \\bowtie
 \\subseteq            \\supseteq           \\cong               \\Join$^b$
 \\sqsubset$^b$        \\sqsupset$^b$       \\neq                \\smile
 \\sqsubseteq          \\sqsupseteq         \\doteq              \\frown
 \\in                  \\ni                 \\propto             =
 \\vdash               \\dashv              <                   >
 :

  ,            ;           :              \\ldotp              \\cdotp

  \\leftarrow                   \\longleftarrow              \\uparrow
 \\Leftarrow                   \\Longleftarrow              \\Uparrow
 \\rightarrow                  \\longrightarrow             \\downarrow
 \\Rightarrow                  \\Longrightarrow             \\Downarrow
 \\leftrightarrow              \\longleftrightarrow         \\updownarrow
 \\Leftrightarrow              \\Longleftrightarrow         \\Updownarrow
 \\mapsto                      \\longmapsto                 \\nearrow
 \\hookleftarrow               \\hookrightarrow             \\searrow
 \\leftharpoonup               \\rightharpoonup             \\swarrow
 \\leftharpoondown             \\rightharpoondown           \\nwarrow
 \\rightleftharpoons           \\leadsto$^b$

  \\ldots               \\cdots              \\vdots              \\ddots
 \\aleph               \\prime              \\forall             \\infty
 \\hbar                \\emptyset           \\exists             \\Box$^b$
 \\imath               \\nabla              \\neg                \\Diamond$^b$
 \\jmath               \\surd               \\flat               \\triangle
 \\ell                 \\top                \\natural            \\clubsuit
 \\wp                  \\bot                \\sharp              \\diamondsuit
 \\Re                  \\|                  \\backslash          \\heartsuit
 \\Im                  \\angle              \\partial            \\spadesuit
 \\mho$^b$             .                   |

  \\sum                 \\bigcap             \\bigodot
 \\prod                \\bigcup             \\bigotimes
 \\coprod              \\bigsqcup           \\bigoplus
 \\int                 \\bigvee             \\biguplus
 \\oint                \\bigwedge

  \\arccos     \\cos       \\csc      \\exp      \\ker         \\limsup      \\min      \\sinh
 \\arcsin     \\cosh      \\deg      \\gcd      \\lg          \\ln          \\Pr       \\sup
 \\arctan     \\cot       \\det      \\hom      \\lim         \\log         \\sec      \\tan
 \\arg        \\coth      \\dim      \\inf      \\liminf      \\max         \\sin      \\tanh

  (                    )                   \\uparrow            \\Uparrow
 [                    ]                   \\downarrow          \\Downarrow
 \\{                   \\}                  \\updownarrow        \\Updownarrow
 \\lfloor              \\rfloor             \\lceil              \\rceil
 \\langle              \\rangle             /                   \\backslash
 |                    \\|

   \\rmoustache        \\lmoustache         \\rgroup            \\lgroup
  \\arrowvert         \\Arrowvert          \\bracevert

   \\hat            \\acute         \\bar           \\dot           \\breve
 \\check          \\grave         \\vec           \\ddot          \\tilde

  \\widetilde                     \\widehat
 \\overleftarrow                 \\overrightarrow
 \\overline                      \\underline
 \\overbrace                     \\underbrace
 \\sqrt                          \\sqrt
                                 \\frac
"""

accent_name_list = [
    "acute", "grave", "hat", "tilde", "check", "breve", "overline", 'dotaccent',
    "dot", "ddot", "vec", "dddot", 'bar', 'vector', 'macron', 'circumflex'
    #"ttilde",
    # not default Latex accent
    #"circumflex"
]

under_name_list = [
    "underline",
    "underbrace",
    #"underarc"
]
def get_latex_commands():
    """
    return list of latex command

    :return:
    """
    cmds = []
    for m in re.finditer(r"\\\w+", latex_math_symbols):
        # print m, m.group()
        cmds.append(m.group()[1:])
    return cmds


big_op_name_list = ["int", "oint", "sum", "prod", "coprod", "bigcup", "bigcap", "bigvee", "bigwedge",
                    "bigoplus", "bigotimes", "iint", "iiint", "iiiint", "idotsint"]


def is_unary_op(name):
    """
    just test whether it's unary,
    if false could either be non-op or binary-op

    :param name:
    :return:
    """
    if name.startswith("\\"):
        name = name[1:]
    return name in ['-', 'minus', 'neg', 'forall', 'exists', 'partial', 'nabla']


symbol_name2set_type = {
    "B": "booleans",
    "Z": "integers",
    "R": "reals",
    "Q": "rationals",
    "N": "naturalnumbers",
    "C": "complexes",
    "P": "primes"
}


unit_list = [
    'rev', 'min', 'rev', 'sec', 'rad', 'kg'
]


digit_english_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']