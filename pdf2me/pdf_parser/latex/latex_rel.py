latex_rel_list = [
    "\\le","\\le", "\\ge","\\ge", "\\neq","\\neq",
    "\\sim","\\sim", "\\ll","\\ll", "\\gg","\\gg",
    "\\doteq","\\doteq", "\\simeq","\\simeq", "\\subset","\\subset",
    "\\supset","\\supset", "\\approx","\\approx", "\\asymp","\\asymp",
    "\\subseteq","\\subseteq", "\\supseteq","\\supseteq", "\\cong","\\cong",
    "\\smile","\\smile", "\\sqsubset","\\sqsubset", "\\sqsupset","\\sqsupset",
    "\\equiv","\\equiv", "\\frown","\\frown", "\\sqsubseteq","\\sqsubseteq",
    "\\sqsupseteq","\\sqsupseteq", "\\propto","\\propto", "\\bowtie","\\bowtie",
    "\\in","\\in", "\\ni","\\ni", "\\prec","\\prec",
    "\\succ","\\succ", "\\vdash","\\vdash", "\\dashv","\\dashv",
    "\\preceq","\\preceq", "\\succeq","\\succeq", "\\models","\\models",
    "\\perp","\\perp", "\\parallel","\\parallel",
    "\\mid","\\mid", "\\bumpeq","\\bumpeq",
    '=', '>', '<',

    # adddtional
    '\\leq', '\\geq',
]


def is_rel_latex_val(latex_val):
    if latex_val.startswith("\\not"):
        return True
    if "arrow" in latex_val:
        return True
    return latex_val in latex_rel_list