latex_punct_list = [
    '\\cdots', '\\ldots', '\\cdot', '\\dot',
    ':', ';', ',', '.',
    '\\colon', '\\semicolon', '\\comma', '\\period',
]


def is_punct_late_val(latex_val):
    return latex_val in latex_punct_list