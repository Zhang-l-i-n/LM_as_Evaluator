from nltk.translate.bleu_score import sentence_bleu
from bert_score import score

reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
score_bleu = sentence_bleu(reference, candidate)
print(score_bleu)

cands = ['天天干家务烦死了', '难受死了啊']
refs = ['这也完全不相干啊', '真的难受死了啊']

P, R, F1 = score(cands, refs, lang="zh", verbose=True)

print(f"System level F1 score: {F1.mean():.3f}")
