import pandas as pd
import re
import os

def convert(file, base_dir=''):
    df = pd.read_excel(file)
    df = df[[df.columns[0], df.columns[1], df.columns[-2], df.columns[-1]]]
    text = df.to_csv(sep='\t')
    sent_id = 0
    
    outp = []
    
    for line in text.splitlines()[1:]:
        if '# sent_id' in line:
            sent_id = re.search("sent_id = ([0-9]+)", line).group(1)
        try:
            line_id, token_id, token, aspect, mark = line.split('\t')
        except:
            continue
        if aspect and mark:
            outp.append([sent_id, (int(token_id), ), aspect, mark])
    
    outp = select_slices(outp)
    
    new_file = os.path.join(base_dir, file.split('.')[0] + '_tonal_words.tsv')
    
    with open(new_file, 'w', encoding='utf-8') as f:
        for sent_id, token_ids, aspect, mark in outp:
            f.write(sent_id)
            f.write('\t')
            f.write(','.join([str(i) for i in token_ids]))
            f.write('\t')
            f.write(aspect)
            f.write('\t')
            f.write(str(int(float(mark))))
            f.write('\n')
    
    print(new_file + ' - ready!')

def select_slices(inp):
    outp = [inp[0]]
    
    for token1, token2 in zip(inp[:-1], inp[1:]):
        sent_id1, token_id1, aspect1, mark1 = token1
        sent_id2, token_id2, aspect2, mark2 = token2
        token_id1 = token_id1[0]
        token_id2 = token_id2[0]
        if sent_id1 == sent_id2 and aspect1 == aspect2 and mark1 == mark2:
            if token_id2 == token_id1 + 1:
                outp[-1][1] = (outp[-1][1][0], token_id2)
            else:
                outp.append(token2)
        else:
            outp.append(token2)
    
    return outp

for f in os.listdir():
    if f.endswith('.xlsx'):
        convert(f, base_dir='разметка_финал')