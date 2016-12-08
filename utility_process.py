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
    
def tokenize(lines, unchanged=False):
    # punctuation etc.
    punkt = ['.', ';', ',', '(', ')', '[', ']', '!', '?', "'", '`', '´', '"', '#', '$', '&']
    names = []
    numbers = []
    links = []
    for i,line in enumerate(lines):
        # initialization
        result = []
        current_token = ""
        # iterate through line
        for i in range(0, len(line)):
            char = line[i]
            # skip newline
            if char == "\n":
                continue
            # handle punctuation
            elif char in punkt:
                # handle ampersand
                if char == ';' & current_token == "amp":
                    result.append("&")
                    current_token = ""
                    continue
                if current_token != "":
                    result.append(current_token)
                result.append([char])
                current_token = ""
            # handle whitespace
            elif char == " ":
                if current_token != "":
                    # handle ampersand
                    if current_token == 'amp':
                        current_token = "&"
                    result.append(current_token)
                current_token = ""
            # handle usernames
            elif char == "@":
                if current_token != "":
                    result.append(current_token)
                result.append("@")
                j = i
                while line[j] != " ":
                    j += 1
                if unchanged:
                    result.append(line[i:j])
                    current_token = ""
                else:
                    result.append("<user")
                    current_token = ""
                    names.append(line[i:j])
                i = j + 1
            # handle links
            elif line[i:i+4] == "http":
                if current_token != "":
                    result.append(current_token)
                j = i
                while line[j] != " ":
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
            elif line[i] in string.digits && current_token == "":
                j = i
                while line[j] in (string.digits + ['.',',']):
                    j += 1
                if unchanged:
                    result.append(line[i:j])
                else:
                    result.append("<number>")
                    numbers.append(line[i:j])
                i = j 
        # end inner for
        lines[i] = result
    # end of outer for
    return (names, numbers, links, lines)