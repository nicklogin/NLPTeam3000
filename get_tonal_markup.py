import json

def load_json(f):
    with open(f, 'r', encoding='utf-8') as inp:
        result = json.load(inp)
    return result

def get_tonal_markup(wordlist1, bigram_list1, trigram_list1, wordlist2, bigram_list2, trigram_list2, aspects, conllu_folder):
  new_path = conllu_folder + '_auto_processed'

  if not os.path.exists(new_path):
    os.mkdir(new_path)

  for file in os.listdir(conllu_folder):
    outp = []
    path = os.path.join(conllu_folder, file)
    t = conllu.parse(open(path, 'r', encoding='utf-8').read())
    for sent_id, sent in enumerate(t):
      sent_id += 1
      t_id1 = 0
      for token1, token2, token3 in zip([i['lemma'] for i in sent], [i['lemma'] for i in sent[1:]]+[''], [i['lemma'] for i in sent[2:]]+['','']):
        t_id1 += 1
        trigram = token1+' '+token2+' '+token3
        bigram = token1+' '+token2
        if trigram in trigram_list1:
          outp.append(str(sent_id)+'\t'+str(t_id1)+','+str(t_id1+2)+'\t'+aspects[0]+'\t'+str(trigram_list1[trigram]))
        elif bigram in bigram_list1:
          outp.append(str(sent_id)+'\t'+str(t_id1)+','+str(t_id1+1)+'\t'+aspects[0]+'\t'+str(bigram_list1[bigram]))
        elif token1 in wordlist1:
          outp.append(str(sent_id)+'\t'+str(t_id1)+'\t'+aspects[0]+'\t'+str(wordlist1[token1]))
        elif trigram in trigram_list2:
          outp.append(str(sent_id)+'\t'+str(t_id1)+','+str(t_id1+2)+'\t'+aspects[1]+'\t'+str(trigram_list2[trigram]))
        elif bigram in bigram_list2:
          outp.append(str(sent_id)+'\t'+str(t_id1)+','+str(t_id1+1)+'\t'+aspects[1]+'\t'+str(bigram_list2[bigram]))
        elif token1 in wordlist2:
          outp.append(str(sent_id)+'\t'+str(t_id1)+'\t'+aspects[1]+'\t'+str(wordlist2[token1]))
    
    path = os.path.join(conllu_folder+"_auto_processed", file[:file.rfind('.')]+"_auto_processed.tsv")
    with open(path, 'w', encoding='utf-8') as file_to_write:
      for line in outp:
        file_to_write.write(line+'\n')


food = load_json('all_food.json')

wordlist1 = {k:v for k,v in food.items() if k.count(' ') == 0}
bigram_list1 = {k:v for k,v in food.items() if k.count(' ') == 1}
trigram_list1 = {k:v for k,v in food.items() if k.count(' ') == 2}

service = load_json('all_service.json')

wordlist2 = {k:v for k,v in service.items() if k.count(' ') == 0}
bigram_list2 = {k:v for k,v in service.items() if k.count(' ') == 1}
trigram_list2 = {k:v for k,v in service.items() if k.count(' ') == 2}

aspects = ['food', 'service']

if name == '__main__':
    get_tonal_markup(wordlist1, bigram_list1, trigram_list1,
                    wordlist2, bigram_list2, trigram_list2,
                    aspects=aspects,
                    conllu_folder='')