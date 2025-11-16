# postprocess.py — (Anushka Future Work)

def fix_spacing(text):
    text = text.replace(" .", ".").replace(" ,", ",")
    return text

def postprocess(text):
    text = fix_spacing(text)
    return text

