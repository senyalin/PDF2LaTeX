# http://artofproblemsolving.com/wiki/index.php?title=LaTeX:Symbols
# one latex value could be both unary and binary

latex_op_list = [
    "\\pm", "\\pm", "\\mp", "\\mp", "\\times", "\\times",
    "\\div", "\\div", "\\cdot", "\\cdot", "\\ast", "\\ast",
    "\\star", "\\star", "\\dagger", "\\dagger", "\\ddagger", "\\ddagger",
    "\\amalg", "\\amalg", "\\cap", "\\cap", "\\cup", "\\cup",
    "\\uplus", "\\uplus", "\\sqcap", "\\sqcap", "\\sqcup", "\\sqcup",
    "\\vee", "\\vee", "\\wedge", "\\wedge", "\\oplus", "\\oplus",
    "\\ominus", "\\ominus", "\\otimes", "\\otimes", "\\circ", "\\circ",
    "\\bullet", "\\bullet", "\\diamond", "\\diamond", "\\lhd", "\\lhd",
    "\\rhd", "\\rhd", "\\unlhd", "\\unlhd", "\\unrhd",
    "\\oslash", "\\oslash", "\\odot", "\\odot", "\\bigcirc", "\\bigcirc",
    "\\triangleleft", "\\triangleleft", "\\Diamond", "\\Diamond", "\\bigtriangleup", "\\bigtriangleup",
    "\\bigtriangledown", "\\bigtriangledown", "\\Box", "\\Box", "\\triangleright", "\\triangleright",
    "\\setminus", "\\setminus", "\\wr", "\\wr",
]
latex_bin_op_list = [
    "\\pm", "\\pm", "\\mp", "\\mp", "\\times", "\\times",
    "\\div", "\\div", "\\cdot", "\\cdot",
    "\\star", "\\star", "\\dagger", "\\dagger", "\\ddagger", "\\ddagger",
    "\\cap", "\\cap", "\\cup", "\\cup",
    "\\uplus", "\\uplus", "\\sqcap", "\\sqcap", "\\sqcup", "\\sqcup",
    "\\vee", "\\vee", "\\wedge", "\\wedge", "\\oplus", "\\oplus",
    "\\ominus", "\\ominus", "\\otimes", "\\otimes", "\\circ", "\\circ",
    "\\diamond", "\\diamond", "\\lhd", "\\lhd",
    "\\rhd", "\\rhd", "\\unlhd", "\\unlhd", "\\unrhd",
    "\\oslash", "\\oslash", "\\odot", "\\odot", "\\bigcirc", "\\bigcirc",
    "\\triangleleft", "\\triangleleft", "\\Diamond", "\\Diamond", "\\bigtriangleup", "\\bigtriangleup",
    "\\bigtriangledown", "\\bigtriangledown", "\\Box", "\\Box", "\\triangleright", "\\triangleright",
    "\\setminus", "\\setminus",
    "+", "-", "*", "/"
]
latex_unary_op_list = [
    "-", "~",
    '\\neg',
    '\\forall', '\\exists',
    '\\partial', '\\nabla',
]

latex_arrow_list = [
    "\\gets","\\to","\\leftarrow","\\Leftarrow","\\rightarrow","\\Rightarrow","\\leftrightarrow","\\Leftrightarrow",
    "\\mapsto","\\hookleftarrow","\\leftharpoonup","\\leftharpoondown","\\rightleftharpoons","\\longleftarrow",
    "\\Longleftarrow","\\longrightarrow","\\Longrightarrow","\\longleftrightarrow","\\Longleftrightarrow",
    "\\longmapsto","\\hookrightarrow","\\rightharpoonup","\\rightharpoondown","\\leadsto","\\uparrow","\\Uparrow",
    "\\downarrow","\\Downarrow","\\updownarrow","\\Updownarrow","\\nearrow","\\searrow","\\swarrow","\\nwarrow"
]

bid_op_list = [
    '\\int', '\\sum', '\\prod', '\\oint', '\\coprod'
]


def is_op_latex_val(latex_val):
    """
    This is a functionality assessment, used in the constraint creation.

    :param latex_val:
    :return:
    """
    return is_unary_op_latex_val(latex_val) or \
           is_bin_op_latex_val(latex_val) or \
           latex_val in bid_op_list


def is_unary_op_latex_val(latex_val):
    """
    This is a functionality assessment, not glyph type assessment
    used in the constraint creation.

    :param latex_val:
    :return:
    """
    return latex_val in latex_unary_op_list


def is_bin_op_latex_val(latex_val):
    """
    This is a functionality assessment, not glyph type assessment
    used in the constraint creation.

    :param latex_val:
    :return:
    """
    return latex_val in latex_bin_op_list


def is_arrow_op_latex_val(latex_val):
    """

    :param latex_val:
    :return:
    """
    return latex_val in latex_arrow_list
