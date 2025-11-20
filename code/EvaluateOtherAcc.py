from code.base_class.evaluate import evaluate
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
class EvaluateOtherAcc(evaluate):
    data = None
    def evaluate(self):
        return roc_auc_score(self.data['true_y'], self.data['pred_y']), average_precision_score(self.data['true_y'], self.data['pred_y'], pos_label=0)
        