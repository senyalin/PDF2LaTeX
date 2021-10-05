
save_images = True

project_path = 'E:/zelun/'
pdf_path = 'E:/zelun/pdfs/'
filename = 'im2markup'
tmp_folder = 'E:\zelun\pdfs/temp_data/'

TIMEOUT = 3

image_resolution = 255
downsample_ratio = 2.0

padding_size = [4, 4, 4, 4]

# smaller images
buckets = [[120,64],[120,160],[240,64],[240,160],[360,64],[360,64],[360,160],
           [480,64],[480,100],[480,160],[480,200],[660,64],[660,100],
           [660,160],[800,100]]

stopwords = set([
    'a','all','and','at','an','as',
    'be','by','both',
    'can',
    'due','do','der',
    'etc','etc.',
    'find','for','flow',
    'gle','get','go','gives',
    'have','hold','holds','has',
    'in','is','it','it,','if','into',
    'law','let','long','less','last','low',
    'not',
    'of','on','only',
    'set','see','spin','such',
    'the','to','true','that','test','taking','thus',
    'up',
    'we','with','will','which',
    '()','(),'
])

# larger images
# buckets = [[120,64],[120,160],[240,64],[240,160],[360,64],[360,100],[360,160],
#            [360,200],[480,64],[480,100],[480,160],[480,200],[660,64],[660,100],
#            [660,160],[800,100],[800,200],[960,64],[960,100]]

# latex_template = r"""
#     \documentclass[12pt]{article}
#     \pagestyle{empty}
#     \usepackage{mathtools}
#     \usepackage{graphicx}
#     \newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
#     \newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
#     \begin{document}
#     \scalebox{2}{$%s$}
#     \end{document}
#     """

latex_template = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{mathtools}
\usepackage{graphicx}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\[
%s
\]
\end{document}
"""

latex_template_plaintext = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{mathtools}
\usepackage{graphicx}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
%s
\end{document}
"""

latex_template_bf = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{mathtools}
\usepackage{graphicx}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\textbf{%s}
\end{document}
"""

latex_template_it = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{mathtools}
\usepackage{graphicx}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\textit{%s}
\end{document}
"""

latex_template_bfif = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{mathtools}
\usepackage{graphicx}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\textit{\textbf{%s}}
\end{document}
"""

latex_template_sec = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{mathtools}
\usepackage{graphicx}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\section{%s}
\end{document}
"""

latex_template_para = r"""
\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{mathtools}
\usepackage{graphicx}
\newcommand{\mymatrix}[1]{\begin{matrix}#1\end{matrix}}
\newcommand{\mypmatrix}[1]{\begin{pmatrix}#1\end{pmatrix}}
\begin{document}
\paragraph{%s}
\end{document}
"""