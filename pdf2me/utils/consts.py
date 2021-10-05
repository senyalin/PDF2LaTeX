from utils.math_resources import func_name_list

PLOT_MODE = False

script_ratio = 0.90  # font size smaller than this ratio is considered sub/super-script

data_folder = 'test_files'

special_nme_glyph_list = ['bullet',  'fi', 'ffi', 'fl',
                          '\xef\xac\x81', 'section', 'daggerdbl']

additional_words = [
    'gcd',
    'i.e.', 'programme', 'lagrangian', 'optimality',
    'empirically', 'multi', 'regularizer', 'preprocessing',
    'two', 'of', 'by', 'whose', 'are', 'the',
    'bag-of-means', 'http', 'www', 'pinyin', 'html', 'model', 'performance',
    'minimum', 'estimator', 'well-known', 'mean-square',
    'summation',
    # punctuation
    '.', ';'
]

chunk_exception_list = ['c++', 'C++']

concatenation_operators = {
        'equal', '=', 'plus', '+', 'minus', 'lessequal', 'greaterequal', 'less', '<',
        'greater', '>', 'element', 'reflexsubset', 'partialdiff', 'arrowright',
        'arrowdblright', 'gradient', 'suchthat', 'intersection', 'union', '\\in',
        'notequal', 'equivalence', 'propersubset', 'propersuperset', 'negationslash',
        'logicaland', 'logicalor'
    }

bullet_list = ['bullet', 'endash', 'square4']

short_math_words = [
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',  # trigonometric function
        'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',  # hyperbolic function
        'exp', 'ln', 'log',
        'det', 'tr', 'diag','gcd', 'mod',
        'avg', 'std', 'calc',
        'min', 'max', 'lim',
        'arg', 'deg', 'dim',
        'Harr', "Jac", 'sgn',
        "ker", "len", "sup",
    ]

math_words = [
    'w.r.t.', 's.t.', 'if', 'res'
]
math_words.extend(func_name_list)

# all treated as math
math_font_key_list = ['MathematicalPi', 'GreekwithMathPi', "+MSBM", "Math"]

alphabeta_font_key_list = ["+CMSY", "+CMMI"]

left_parentheses = ['[', '{', '(']
right_parentheses = [']', '}', ')']