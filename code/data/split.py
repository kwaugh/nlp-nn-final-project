from pickle import load, dump
import random

print("Starting splitting")
en_lines = []
with open('en.pkl', 'rb') as en_f:
    en_lines = load(en_f)

fr_lines = []
with open('fr.pkl', 'rb') as fr_f:
    fr_lines = load(fr_f)

idxs = [i for i in range(len(en_lines))]
train_idxs = random.sample(idxs, int(0.8 * len(idxs)))

train_idxs_set = set(train_idxs)
idxs = [x for x in idxs if x not in train_idxs_set]

valid_idxs = random.sample(idxs, int(0.5 * len(idxs)))

valid_idxs_set = set(valid_idxs)
test_idxs = [x for x in idxs if x not in valid_idxs_set]

with open('train.pkl', 'wb') as f:
    zipped = [(en_lines[i], fr_lines[i]) for i in train_idxs]
    dump(zipped, f)

with open('valid.pkl', 'wb') as f:
    zipped = [(en_lines[i], fr_lines[i]) for i in valid_idxs]
    dump(zipped, f)

with open('test.pkl', 'wb') as f:
    zipped = [(en_lines[i], fr_lines[i]) for i in test_idxs]
    dump(zipped, f)
