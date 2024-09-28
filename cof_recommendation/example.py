from recommendation import *

POINT = ('ald_1', 'amine_11')
EVALUATION = 1

cofr = CofRecommendation()
cofr.register(POINT, EVALUATION)
batch, _ = cofr.suggest_batch()
print(batch)
