# -*- coding: utf-8 -*-

import random
import string
      
# postprocess tweets, fixing hashtags, links etc.       
def postprocess(seq, usernames, numbers, links):
    # cutoff start and end
    seq = seq[1:-1]  
    # replace names, numbers, links
    for pos, word in enumerate(seq):
        if word == "<user>":
            seq[pos] = random.choice(usernames)
        elif word == "<number>":
            seq[pos] = random.choice(numbers)
        elif word == "<link>":
            seq[pos] = random.choice(links)
    # join to string
    seq = " ".join(seq)
    # postprocess some spcial words
    seq = seq.replace("@ ", "@")
    seq = seq.replace("# ", "#")
    seq = seq.replace(" 's", "'s")
    seq = seq.replace(" 're", "'re")
    seq = seq.replace(" n't", "n't")
    seq = seq.replace(" 'm", "'m")
    seq = seq.replace(" !", "!")
    seq = seq.replace(" .", ".")
    seq = seq.replace(" ?", "?")
    return seq
    
def tokenize(lines, unchanged=False, case=False):
    # punctuation etc.
    punkt = ['.', ';', ':', ',', '(', ')', '[', ']', '!', '?', "'", '"', '#', '$', '&', u'´', u'`', '-']
    names = []
    numbers = []
    links = []
    for k,line in enumerate(lines):
        # initialization
        result = []
        current_token = ""
        # iterate through line
        i = 0
        while i < len(line):
            char = line[i]
            # end on newline
            if char == "\n":
                if current_token != "":
                    result.append(current_token)
                i += 1
            # handle punctuation
            elif char in punkt:
                # handle ' inside words
                if (char == "'") and (current_token != "") & (line[i+1] != " "):
                    current_token += char
                    i += 1
                # handle ampersand
                elif (char == ';') and (current_token == "amp"):
                    result.append("&")
                    current_token = ""
                    i += 1
                # handle --
                elif (char == '-') and (line[i+1] == '-'):
                    if current_token != "":
                        result.append(current_token)
                    result.append("--")
                    current_token = ""
                    i += 2
                # handle abbreviations with dots and ...
                elif (char == '.') and (line[i+1] in (string.ascii_letters + '.')):
                    j = i
                    while (j < len(line)) and (line[j] in (string.ascii_letters + '.')):
                        j += 1
                    current_token += line[i:j]
                    result.append(current_token)
                    current_token = ""
                    i = j + 1
                # append token and symbol
                else:
                    if current_token != "":
                        result.append(current_token)
                    result.append(char)
                    current_token = ""
                    i += 1
            # handle whitespace
            elif char == " ":
                if current_token != "":
                    # handle ampersand
                    if current_token == 'amp':
                        current_token = "&"
                    result.append(current_token)
                current_token = ""
                i += 1
            # handle usernames
            elif char == "@":
                if current_token != "":
                    result.append(current_token)
                result.append("@")
                j = i
                while (j < len(line)) and (line[j] != " "):
                    j += 1
                if unchanged:
                    result.append(line[i:j])
                    current_token = ""
                else:
                    result.append("<user>")
                    current_token = ""
                    names.append(line[i:j])
                i = j + 1
            # handle links
            elif (i+4 <= len(line)) and (line[i:i+4] == "http"):
                if current_token != "":
                    result.append(current_token)
                j = i
                while (j < len(line)) and (line[j] != " "):
                    j += 1
                if unchanged:
                    result.append(line[i:j])
                    current_token = ""
                else:
                    result.append("<link>")
                    current_token = ""
                    links.append(line[i:j])
                i = j + 1
            # handle numbers
            elif (line[i] in string.digits) and (current_token == ""):
                j = i
                while (j < len(line)) and (line[j] in (string.digits + '.,')):
                    j += 1
                if unchanged:
                    result.append(line[i:j])
                else:
                    result.append("<number>")
                    numbers.append(line[i:j])
                i = j 
            # handle normal stuff
            else:
                current_token += char
                i += 1
        # end inner for
        # handle case
        if not case:
            result = [w.lower() for w in result]
        lines[k] = result
    # end of outer for
    return (names, numbers, links, lines)