from recommendation import *

# batch_1
POINTS = [
    ('ald_1', 'amine_11'),
    ('ald_7', 'amine_11'),
    ('ald_8', 'amine_12'),
    ('ald_20', 'amine_11'),
    ('ald_13', 'amine_8')
]
EVALUATIONS = [0, 0, 0, 0, 1]

cofr = CofRecommendation()
for i in range(len(POINTS)):
    cofr.register(POINTS[i], EVALUATIONS[i])

# batch_2
POINTS = [
    ('ald_13', 'amine_11'),
    ('ald_13', 'amine_12'),
    ('ald_13', 'amine_10'),
    ('ald_25', 'amine_12'),
    ('ald_25', 'amine_10'),
    ('ald_23', 'amine_12')
]
EVALUATIONS = [2, 2, 1, 2, 0, 2]

for i in range(len(POINTS)):
    cofr.register(POINTS[i], EVALUATIONS[i])

batch = [
    ('ald_11', 'amine_11'),
    ('ald_8', 'amine_11'),
    ('ald_8', 'amine_17'),
    ('ald_10', 'amine_11'),
    ('ald_21', 'amine_11'),
    ('ald_13', 'amine_15'),
    ('ald_24', 'amine_11'),
    ('ald_25', 'amine_11'),
    ('ald_25', 'amine_17'),
    ('ald_25', 'amine_8'),
    ('ald_23', 'amine_11'),
]

cofr.evaluate_batch(batch)
