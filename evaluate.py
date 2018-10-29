'''
Run in nlp env: `source activate nlp`
'''
import sys
from gensim.models.doc2vec import Doc2Vec

def load_model(path):
    return Doc2Vec.load(path)

def vec(modelfp, word):
    return load_model(modelfp).wv[word]

def similar(modelfp, word):
    wv = load_model(modelfp).wv
    return wv.most_similar(word)

def similar2(modelfp, positives, negtives):
    wv = load_model(modelfp).wv
    if negtives is None:
        print('no negtives.')
        result = wv.most_similar(positive=positives)
    else:
        print('has negtives.')
        result = wv.most_similar(positive=positives, negative=negtives)
    for x in result:
        print("{}: {:.4f}".format(*x))

def doc_similar2(modelfp, positives, negtives):
    model = load_model(modelfp)
    if negtives is None:
        print('no negtives.')
        result = model.docvecs.most_similar([model.infer_vector(positives)])
    else:
        print('has negtives.')
        result = model.docvecs.most_similar(positive=[model.infer_vector(positives)], negative=[model.infer_vector(negtives)])
    for x in result:
        print("{}: {:.4f}".format(*x))

def main():
    print(sys.argv)
    model_filep = sys.argv[2]
    word = sys.argv[3]
    if sys.argv[1] == 'vec':
        print('vec: {}'.format(vec(model_filep, word)))

    elif sys.argv[1] == 'simi':
        print('similars: {}'.format(similar(model_filep, word)))

    elif sys.argv[1] == 'simi2':
        positives = word.split(" ")
        negtives = sys.argv[4].split(" ") if len(sys.argv) > 4 else None
        print('get poss: %s, negs: %s' % (positives, negtives))
        similar2(model_filep, positives, negtives)

    elif sys.argv[1] == 'dsimi2':
        positives = word.split(" ")
        negtives = sys.argv[4].split(" ") if len(sys.argv) > 4 else None
        print('get poss: %s, negs: %s' % (positives, negtives))
        doc_similar2(model_filep, positives, negtives)

    else:
        print('Usage: tail -n 50 evaluate.py')

if __name__ == '__main__':
    main()
