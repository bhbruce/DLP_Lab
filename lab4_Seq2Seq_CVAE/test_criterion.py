"""============================================================================
example input of Gaussian_score

words = [['consult', 'consults', 'consulting', 'consulted'],
['plead', 'pleads', 'pleading', 'pleaded'],
['explain', 'explains', 'explaining', 'explained'],
['amuse', 'amuses', 'amusing', 'amused'], ....]

the order should be : simple present, third person, present progressive, past
============================================================================"""
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = './dataset/train.txt'  # should be your directory of train.txt
    with open(yourpath, 'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            # print(word)
            words_list.extend([word])
        for t in words:
            # print(t)
            for i in words_list:
                if t == i:
                    print(i)  # print correct answer
                    score += 1
    return score/len(words)


# compute BLEU-4 score
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33, 0.33, 0.33)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu([reference],
                         output, weights=weights,
                         smoothing_function=cc.method1)
