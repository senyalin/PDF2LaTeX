#############
# the name to latex code
#############
import string
from utils.math_resources import accent_name_list, under_name_list, \
    big_op_name_list, get_latex_commands


pdf_gn2latex = {
}

name2latex = {}


def get_greek_name_list():
    """
    all greek name need add backslash
    """
    from pdf_parser.latex.latex_glyph import xyz_greek_list, xy_greek_list, yz_greek_list, y_greek_list
    greek_list = []
    greek_list.extend(xy_greek_list)
    greek_list.extend(xyz_greek_list)
    greek_list.extend(yz_greek_list)
    greek_list.extend(y_greek_list)
    greek_name_list = [s[1:] for s in greek_list]
    return greek_name_list


name_add_slash_list = [
    "sqrt", "partial",
    # hat
    "tilde", "overline",
    # op
    "wedge", "cap", "oplus", "times", "rtimes", "otimes",
    # relation
    #"leq",
    "in", "geq", "preceq",
    #"notsubset",
    "supseteq", "gg",
    #"supsetnoteqq",
    "subset",
    "leqq", "ni", "simeq", "notin", "supset", "geqq", "succ", "succeq",
    "nsubseteq", "nsupseteq",   # this is available in asymsymb.
    "cong", "prec",
    # greek list
    "alpha", "rho",
    # fence,
    "langle", "rangle", "lfloor", "rfloor",
    # logical
    "forall", "exists", "neg",
    # unknown
    "nmid",
    # arrow
    "Leftrightarrow", "leftarrow", "rightarrow", "hookrightarrow", "Rightarrow", "mapsto",

    # speical
    "emptyset", "infty",
    "pitchfork",  # https://tex.stackexchange.com/questions/365664/why-doesnt-pitchfork-symbol-work

    # punct
    "cdot", "ldots", "cdots", "ast", "square", "bullet", "#",
    
    
    # http://ftp.cvut.cz/tex-archive/macros/latex/contrib/wasysym/wasysym.pdf
    "Join", "Box", "Diamond", "leadsto", "sqsubset", "sqsupset", "lhd", "unlhd", "LHD", "rhd", "unrhd",
    "RHD", "apprle", "apprge", "wasypropto", "invneg", "ocircle", "logof",

    # logic
    "because", "therefore",



]


name_add_slash_list.extend(get_greek_name_list())
name_add_slash_list.extend(accent_name_list)
name_add_slash_list.extend(under_name_list)
name_add_slash_list.extend(big_op_name_list)

name_exact_list = []
name_exact_list.extend(string.lowercase)
name_exact_list.extend(string.uppercase)
name_exact_list.extend(string.digits)
# fill a to z
# fill A to z
name_exact_list.extend([
    "@", '!',
    '-', '+', '*', '/', '|',
    '=', '>', '<',
    ',', '.', ';', ':',
    '(', ')', '[', ']',
    '{', '}',  # TODO, though they are not writen as so in LaTex
])
# TODO
name_exact_list.extend(['fi', 'fl', 'ff'])  # just very strange that 'fi' is in Math


name_mapped = {
    # punct
    "minus": "-",
    "plus": "+",
    "slash": "/",
    'plus-or-minus': "\\pm",
    'minus-or-plus': "\\mp",
    "divide": "/",


    "vert": "|",
    "sim": "~",
    "prime": "'",
    'dprime': "''",
    "quotesingle": "'",
    "perp": "\\perp",

    "fractionalLine": "\\frac",
    "ContinuedFraction": "\\frac",
    "exclamation": "!",

    "varpitchfork": "\\pitchfork",
    "doubleprime": "\\dprime", # in the unicode-math package
    "notequiv": "\\not\\equiv",
    "notsubset": "\\not\\subset",
    "subsetnoteqq": "\\nsubseteq",
    "supsetnoteqq": "\\nsupseteq",
    "SingleEndQuartation": ",", # as far as I could tell

    # TODO, unsure about the exact value
    'daggerdbl': "\\ddagger",
    "quotedbl": "\"",

    "latticetop": "\\top",  #TODO, not sure yet,

    'greatermuch': "\\gg",
    'much-greater-than': "\\gg",
    'lessmuch': "\\ll",
    'much-less-than': "\\ll",
    'defines': "\\triangleq",

    "exclam": "!",
    "underscore": "\\_",
    "numbersign": '#',
    "dollar": "$",
    "asterisk": "\\ast",
    "hatwide": "\\hat",
    "dieresis": "\\\"",
    "macron": "\\=",

    # NOTE, meet the following problem when processing PDF file
    "Scheck": "S", # kind of euro symbols
    "Scedil": "S",
    "eacute": "e",
    "aleph": "a",

    "epsilon1": "\\epsilon",
    "space": "\\ ",

    "negationslash": "\\not",  # NOTE, this might be combined with others, such as the not eq, not divide, etc.

    '*': "\\ast",

    # TODO
    "Rfractur": "R",
    "follows": "\\vdash",

    'circumflex': "\\hat",

    "quotedblleft": "\"",
    "quotedblright": "\"",
    "endash": "-",
    "multiply": "\\times",
    "openbullet": "\\textopenbullet",
    "asteriskmath": "\\ast",

    "emdash": '-',
    'uniontext': "\\cup",


    "circleasterisk": "\\circ\\ast",
    "circledivide": "\\oslash",
    "circlemultiplytext": "\\otimes",
    "empty": "\\emptyset",
    "omicorn": "\\omicorn",

    "MasculineOrdinalIndicator": "\\degree", # https://en.wikipedia.org/wiki/Degree_symbol
    "degree": "\\degree",

    #'maximum': '\\ma'

}

arrow_name_mapped = {

    "arrowdblleft": "\\leftarrow",
    "arrowdblright": "\\rightarrow",

    "arrowleft": "\\leftarrow",
    "arrowright": "\\rightarrow",

    "arrowhookleft": "\\hookleftarrow",
    "arrowhookright": "\\hookrightarrow",

    "arrowdblboth": "\\leftrightarrow",
    "arrowboth": "\\leftrightarrow",

    "arrowup": "\\uparrow",
    "arrowdblup": "\\uparrow",

    "arrowdown": "\\downarrow",
    "arrowdbldown": "\\downarrow",  # TODO, double line

    "dblarrowheadright": "\\rightarrow",
    "dblarrowheadleft": "\\leftarrow",

    "arrownortheast": "\\nearrow",
    "arrownorthwest": "\\nwarrow",
    "arrowsoutheast": "\\searrow",
    "arrowsouthwest": "\\swarrow",

}
name_mapped.update(arrow_name_mapped)

punct_name_mapped = {
    'dash': '-',
    "at": "@",
    "question": "?",
    "ampersand": "&",

    "periodcentered": "\\cdot",
    "period": '.',

    "colon": ":",
    "hyphen": "-",
    "longHyphen": "-",

    "semicolon": ";",
    "comma": ",",
    "percent": "%",
    "sharp": "\\#",

    # TODO,
    "exclamdown" : "!",
    "quotedblbase": "\"",
}
name_mapped.update(punct_name_mapped)


logic_name_mapped = {
    "logicalanddisplay": "\\wedge",
    "logicalandtext": "\\wedge",
    "logicaland": "\\wedge",
    "logicalor": "\\vee",
    "logicalordisplay": "\\vee",
    "logicalortext": "\\vee",
    "logicalnot": "\\neg",
    "universal": "\\forall",
    "existential": "\\exists",
    "turnstileleft": "\\vdash",
    "notturnstile": "\\not\\vdash",
    "forces": "\\models",
}
name_mapped.update(logic_name_mapped)


set_name_mapped = {
    "unionsqdisplay": "\\union",
    "unionmulti": "\\union",
    "union": "\\union",
    "uniondisplay": "\\bigcup",
    "unionsq": "\\union",

    "intersectionsq": "\\cap",
    "intersectiondisplay": '\\bigcap',
    "intersectiontext": '\\bigcap',
    "intersection": "\\cap",
    "element": "\\in",
    "element-of": "\\in",
    "empty": "\\empty",


    # TODO
    "subsetnoteql": "\\subset",
    "subsetsqequal": "\\subseteq",

    # not sure of the meaning of reflex
    "reflexsuperset": "\\supset",
    "reflexsubset": "\\subset",

    # forget proper, not equal?
    "propersubset": "\\subset",
    "propersuperset": "\\supset",

    "almost-equals": '=', # not sure which one
}
name_mapped.update(set_name_mapped)

calculus_name_mapped = {
    "integraltext": "\\int",
    'integraldisplay': "\\int",
    "integral": "\\int",
    "partialdiff": "\\partial",
    "gradient": "\\nabla",
}
name_mapped.update(calculus_name_mapped)

arithmic_name_mapped = {
    "infinity": "\\infty",

    "notequal": "\\neq",
    "identical-to": "\\equiv",
    "not-identical-to": "\\not\\equiv",
    "equals": '=',
    "equal": "=",
    "similarequal": "\\simeq",
    "equal1": "=",
    "eq": "=",

    "greater": ">",
    'greater-than-or-equals': '\\ge',
    'ge': "\\ge",
    'greater-than': '>',
    "greaterorequalslant": "\\ge",
    "greaterequal": "\\ge",

    "lessorequalslant": "\\le",
    "leq": "\\le",
    'le': '\\le',
    "less": "<",
    'less-than': "<",
    'less-than-or-equals': "\le",
    "precedesorcurly": "\\prec",
    "lessornotdbleql": "\\le",
    "lessequal": "\\le",
    "similar": "\\sim",
    "approxorequal": "\\cong",
    "equivalence": "\\equiv",

    "precedes": "\\prec",

    "radicalBigg": "\\sqrt",
    "radicalbig": "\\sqrt",
    "radicaltp": "\\sqrt",
    "radicalbt": "\\sqrt",
    "radical": "\\sqrt",

    "greaterorsimilar": "\\ge",
    "lessorsimilar": "\\le",
    "equivasymptotic": "\\sim",
    "followsequal": "=",

    "summationtext": "\\sum",
    "producttext": "\\prod",
    "summation": '\\sum',
    "product": "\\prod",
    "productdisplay": '\\prod',
    'summationdisplay': "\\sum",

    "vector": "\\vec",
    "vextendsingle": "\\vec",
    "radicalbigg": "\\sqrt",
    "radicalBig": "\\sqrt",

    "fraction": "/",
    "slashbig": "/",
    "slashBig": "/",
    "slash": "/",
    "plusminus": "\\pm",
    "dotmath": "\\dot",
    "circledot": "\\odot",
    "circledottext": "\\odot",
    "circledotdisplay": "\\odot",
    "circleplus": "\\oplus",
    "circleplusdisplay": "\\oplus",
    "circleplustext": "\\oplus",
    "circlemultiply": "\\otimes",
    "proportional": "\\propto",
    "proportional-to": "\\propto",
    "precedesequal":  "\\preceq",
    "approxequal": "\\approx",

    'dimension': 'dim',
    'determinant': 'det',

    #absolute-value, should not call name 2 latex
}
name_mapped.update(arithmic_name_mapped)

algebra_name_mapped = {
    "intercal": "\\intercal",
}
name_mapped.update(algebra_name_mapped)

geometry_name_mapped = {
    "perpendicular": "\\perp"
}
name_mapped.update(geometry_name_mapped)

decorator_mapped = {
    "hatwidest": "\\hat",
    "tildewider": "\\tilde",
    "hatwider": "\\hat",
    "tildewide": "\\tilde",
    "asciitilde": "\\tilde",
}
accent_dececorator_list = [
    "widest",
    "winder",
]
for accent_name in accent_name_list:
    for dec in accent_dececorator_list:
        decorator_mapped[accent_name+dec] = "\\"+accent_name

name_mapped.update(decorator_mapped)

normal_fence_mapped = {

    # brace
    "braceleftBig": "{",
    "bracerightBig": "}",
    "bracehtipupleft": "{",
    "bracehtipupright": "}",
    "bracehtipdownleft": "{",
    "bracehtipdownright": "}",
    "braceleftbt": "{",
    "bracerightbt": "}",
    "braceleftbigg": "{",
    "braceleftBigg": "{",
    "bracerightbigg": "}",
    "bracerightBigg": "}",
    "braceleft": "{",
    "braceright": "}",
    "braceleftbig": "{",
    "bracerightbig": "}",
    "bracelefttbt": "{",
    "bracerighttbt": "}",
    "braceleftmid": "{",
    "bracerightmid": "}",
    "bracelefttp": "{",
    "bracerighttp": "}",

    # square bracket
    "bracketleftbt": "[",
    "bracketrightbt": "]",
    "bracketleftex": "[",
    "bracketrightex": "]",
    "bracketlefttp": "[",
    "bracketrighttp": "]",
    "bracketleft": "[",
    "bracketright": "]",
    'bracketrightBig': ']',
    'bracketleftBig': '[',
    'bracketleftbigg': '[',
    'bracketrightbigg': ']',
    'bracketleftbig': "[",
    'bracketrightbig': "]",
    'bracketleftbg': "[",
    'bracketrightbg': "]",
    'largellbracket': '[',
    'largerrbracket': ']',
    'llbracket': '[',
    'rrbracket': ']',
    'bracketleftBigg': '[',
    'bracketrightBigg': ']',

    # parenthesis
    "parenleftbig": "",
    "parenrightbig": ")",
    "parenleft": "(",
    "parenright": ")",
    "parenleftbigg": "(",
    "parenrightbigg": ")",
    "parenlefttp": '(',
    "parenrighttp": ')',
    'parenleftBig': "(",
    'parenrightBig': ")",
    "LeftPar": "(",
    "RightPar": ")",
    "parenrightBigg": ")",
    "parenleftBigg": "(",
    "BigRightPar": ")",
    "BigLeftPar": "(",
    "MiddleRightPar": "}",
    "MiddleLeftPar": "{",
    "parenleftex": "(",
    "parenrightex": ")",
    "parenleftbt": "(",
    "parenrightbt": ")",

}
name_mapped.update(normal_fence_mapped)

fence_mapped = {
    "bardbl": "||",  # double vertical bar

    "quoteleft": "\"",
    "quoteright": "\"",

    "floorleftbigg": "\\lfloor",
    "floorrighttbigg": "\\rfloor",
    "floorrightbigg": "\\rfloor",
    "floorleft": "\\lfloor",
    "floorright": "\\rfloor",
    "floorleftBig": "\\lfloor",
    "floorrightBig": "\\rfloor",

    "ceilingrightBig": "\\rceil",
    "ceilingleftBig": "\\lceil",
    "ceilingleft": "\\lceil",
    "ceilingright": "\\rceil",

    "ceilleft": "\\lceil",
    "ceilright": "\\rceil",
    "ceilingleftbig": "\\lceil",
    "ceilingrightbig": "\\rceil",
    "ceilingleftbigg": "\\lceil",
    "ceilingrightbigg": "\\rceil",
    "ceilingleftBigg": "\\lceil",
    "ceilingrightBigg": "\\rceil",

    'angbracketleftbigg': "\\langle",
    'angbracketrightbigg': "\\rangle",
    'angbracketleftBigg': "\\langle",
    'angbracketrightBigg': "\\rangle",
    'angbracketleftbig': "\\langle",
    'angbracketrightbig': "\\rangle",
    'angbracketleft': "\\langle",
    'angbracketright': "\\rangle",
    'angbracketleftBig': "\\langle",
    'angbracketrightBig': "\\rangle",
}
name_mapped.update(fence_mapped)

symbol_name_mapped = {
    "Delta1": "\\Delta",
    "dotlessi": "i",

    "Phi1": "\\Phi",
    "phi1": "\\phi",
    "Psi1": "\\Psi",

    "rho1": "\\rho",
    "Sigma1": "\\Sigma",
    "sigma1": "\\sigma",
    "theta1": "\\theta",

}
name_mapped.update(symbol_name_mapped)

# https://tex.stackexchange.com/tags/accents/info
accent_name2latex = {
    'grave': '\\`',
    'acute': "\\'",
    'circumflex': "\\hat",
    'umlaut': '\\ddot',
    'dieresis': '\\ddot',
    'hungarumlaut': '\\ddot',
    'tilde': "\\tilde",
    'macron': "\\bar",
    'breve': "\\check",
    "cedilla": "\\c",
    'ogonek': "\\k"
}
name_mapped.update(accent_name2latex)
named_modifier_list = [
    'accute',
    'acute',
    'breve',
    'caron',
    'cedilla',
    'check',
    'circumflex',
    'dieresis',
    'dot',
    'grave',
    'fractur',
    'macron',
    'ogonek',
    'ring',
    'slash', # some european? lslash, Lslash,
    'tilde',

]
for c in string.letters:
    for named_modifier in named_modifier_list:
        name_mapped[c+named_modifier] = c

todo_name_mapped = {
    'lslash': 'l',  # might be some european name
    'followsorcurly': "\\ge",

    "squaresolid": "\\blacksquare",
    "diamondmath": '\\diamondsuit',

    "vextenddouble": "\\vec",
    "triangleinv": "\\bigtriangledown",
    "contintegraltext": "\\int",

    "onequarter": "0.25",
    "onehalf": "0.5",
    "minute": "'",  # minute, the degree?

    "dotaccent": "\\dot",
    "dieresis": "\\''",
    "copyright": "\\copyright",

    "greaterornotdbleql": "\\ge",
    "bullet5": "\\bullet",

    "square4": "\\blacksquare",
    "notbar": "\\nmid",
    "sterling": "\\pounds",
    "turnstileleft": "\\vdash",

    "braceex": "\\bar",  # the bar of long brace
    "eth": "D",  # https://en.wikipedia.org/wiki/Eth

    "egrave": "E",  # https://en.wikipedia.org/wiki/%C3%88
    "Ntilde": "N",
    'odieresis': "o",
    "aacute": "a",

    "squaremutiply": "\\square",  # end of section
    "squaremultiply": "\\square",  # end of section

    "thorn": "\\thorn",
    "Thorn": "\\Thorn",

    "radicalvertex": "\\sqrt",

    # http://ftp.cvut.cz/tex-archive/macros/latex/contrib/wasysym/wasysym.pdf
    "apprge": "\\apprge", # approximate greater equal

    "ellipsis": "ldots",
    "precedesorcurly": "\\prec",
    "caron": "\\check",

    "asciicircum": "\\hat",

    # the IM2Latex dataset
    'lscript': "\\mathcal{l}",
    'owner': "\\ni",
    'arrowrighttophalf': "\\rightharpoonup",  # 56600.pdf, todo, check the latex code
    'suppress': "\\", # backslash before the symbols, 64902, 34001
    'pi1': "\\varpi",
    'slurbelow': '\\smile',
    'circleminus': "\\ominus",
    "unionsqtext": "\\sqcup",
    "contintegraldisplay": "\\oint",
    "arrowleftbothalf": "\\leftharpoondown",
    "slashBigg": "/",
    "slashbigg": "/",
    "circlecopyrt": "\\copyright",
    "circlemultiplydisplay": "\\otimes",
    "coproduct": "\\coprod",
    "coproducttext": "\\coprod",

    # TODO
    "arrowbt": "arrowbt",
    "arrowtp": "arrowtp", # 80496
    "turnstileright": "turnstileright", # 85233

}
name_mapped.update(todo_name_mapped)

digit_map = {
    "zero": '0',
    "one": '1',
    "two": '2',
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9"
}

name2latex.update(name_mapped)
name2latex.update(digit_map)
name2latex.update(pdf_gn2latex)

latex_commands = get_latex_commands()
name_add_slash_list.extend(latex_commands)

for name_add_slash in name_add_slash_list:
    name2latex[name_add_slash] = "\\" + name_add_slash
for name_exact in name_exact_list:
    name2latex[name_exact] = name_exact
