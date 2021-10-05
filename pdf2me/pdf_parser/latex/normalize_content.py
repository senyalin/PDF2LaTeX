"""
The normalization is required because some content are written in unicode for the greek
"""
import string

from utils.math_resources import unicode2latex, special_unicode_chars


def normalize_content(content):
    if isinstance(content, str):
        content = content.decode('utf-8', 'ignore')

    res = ""
    for c in content:
        if c in string.printable:
            res += c
        else:
            if c in unicode2latex:
                res += unicode2latex[c]
            elif c in special_unicode_chars:
                res += c
            else:
                print c
                print "single uval unicode {}".format(c.encode('raw_unicode_escape'))
                raise Exception("unknown unicode")
    return res