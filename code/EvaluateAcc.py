'''
Concrete Evaluate class for a specific evaluation metrics
'''


from code.base_class.evaluate import evaluate
from sklearn.metrics import f1_score
from collections import Counter

class EvaluateAcc(evaluate):
    data = None
    def evaluate(self):
        return f1_score(self.data['true_y'], self.data['pred_y'], pos_label=0, average='binary')
        