from utils.math_resources import accent_name_list, under_name_list

y_list = 'acemnorsuvwxz'
xy_list = 'bdhkl'+'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
yz_list = 'gpqy'
# for certain font, the glyph type of 'f' could be xyz
xyz_list = ''+'f'
hxy_list = 'it'  # half x, not to the full extend
hxyz_list = 'j'

xy_greek_list = [
    #'\\Alpha', '\\Beta', -> A, B
    '\\Gamma', '\\Delta',
    #'\\Epsilon', -> E
    #'\\Zeta', -> Z
    #'\\Eta', -> H
    '\\Theta',
    #'\\Iota', -> I
    #'\\Kappa', -> K
    "\\Lambda",
    #'\\Mu', -> M
    #'\\Nu', -> N
    '\\Xi',
    #'\\Omicron', -> O
    "\\Pi",
    #'\\Rho', -> P
    '\\Sigma',
    #'\\Tau', -> T
    '\\Upsilon', '\\Phi',
    #'\\Chi', -> X
    '\\Psi', '\\Omega',
    '\\delta', '\\theta', "\\lambda"
]

y_greek_list = [
    '\\alpha', '\\epsilon', '\\varepsilon',
    '\\iota', '\\kappa',
    '\\nu',
    #'\\omicron', -> o
    "\\pi",
    '\\sigma', '\\tau', '\\upsilon', '\\omega'
]
xyz_greek_list = [
    '\\beta', '\\zeta', '\\xi',
    '\\phi','\\psi', # based on the MS typing, find that mis place them into the yz group
]
yz_greek_list = [
    '\\gamma', '\\eta', '\\mu',  '\\rho',
    '\\chi',  '\\varphi'
]

greek_list = []
greek_list.extend(xy_greek_list)
greek_list.extend(y_greek_list)
greek_list.extend(xyz_greek_list)
greek_list.extend(yz_greek_list)

greek_name_list = [greek_latex[1:] for greek_latex in greek_list]


def is_greek(c):
    if c in greek_list or "\\"+c in greek_list:
        return True
    return False


def is_accent(v):
    if v.startswith('\\'):
        v = v[1:]
    return v in accent_name_list or v in under_name_list