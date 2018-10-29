# coding:utf8
'''
Run in nlp env: `source activate nlp`
'''
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import sys
import json
import jieba
import time

INVALID_PUNCS = (',', '，', '.', '。', '?', '？', '!', '！', '、',
        '...', ' ', '',
        '的')

def load_sens(filep):
    print('load sens from: {}'.format(filep))
    sens = []
    with open(filep, 'r', encoding='utf8') as r:
        lines = r.readlines()
        for line in lines:
            datajson = json.loads(line.strip())
            sen = []
            for k in datajson:
                if k in ('hardLevel', 'cookingTime', 'materialsAmount'):
                    continue

                if type(datajson[k]) == list:
                    for x in datajson[k]:
                        sen.extend([xi for xi in jieba.cut(x) if xi not in INVALID_PUNCS])
                elif type(datajson[k]) == str or type(datajson[k]) == unicode:
                    sen.extend([xi for xi in jieba.cut(datajson[k]) if xi not in INVALID_PUNCS])
            sens.append(sen)

    return sens

def load_model(path):
    return Doc2Vec.load(path)

def train(model_path, sens, init=False):
    print('start training model: {}, len(sens): {}'.format(model_path, len(sens)))

    modelfp = model_path.split('_')[0] + '_' + time.strftime('%y%m%d%H%M')
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sens)]

    with open('data.txt', 'w') as w:
        for i, doc in enumerate(sens):
            w.write('%s, %s\n' % (i, doc))
        print('save doc to data.txt')

    if init:
        model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)
        model.save(modelfp)
    else:
        model = load_model(model_path)
        model.build_vocab(documents, update=True)
        model.train(documents, total_examples=len(documents), epochs=1)
        model.save(modelfp)

    print('save model to: {}'.format(modelfp))

def main():
    if sys.argv[1] == 'train':
        train_filep = sys.argv[2]
        init = sys.argv[3] == 'true' if len(sys.argv) > 3 else False
        pre_model_filep = sys.argv[4] if len(sys.argv) > 4 else 'model/d2v.model'

        sens = load_sens(train_filep)
        train(pre_model_filep, sens, init)
    else:
        print('tail -n 20 d2v.py')

if __name__ == '__main__':
    main()
