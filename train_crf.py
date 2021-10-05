from sklearn_crfsuite import CRF, metrics
import pickle
import os
from sklearn.metrics import average_precision_score, recall_score
from utils.crf_utils import ContionalRandomField


path = 'CRF_data/'
files = [f for f in os.listdir(path) if f[-1] == 't']
files = [f[:-4] for f in files]
TEST = []
PRED = []
for testfile in files:
    # testfile = '86'
    trainfile = [f for f in files if f != testfile]
    # trainfile = files
    # columnProfile.drawWords()

    # # validate grount truth
    # with open(path + testfile + '.pkl', 'rb') as f:
    #     columnProfile = pickle.load(f)
    # f = open(path + testfile + ".txt", "r")
    # gt = f.readlines()
    # gt = [line[:-1].split(' ') for line in gt]
    # columnProfile.drawLabel(gt)

    # prepare the training data
    x_train = []
    y_train = []
    for filename in trainfile:
        # read data
        with open(path + filename + '.pkl', 'rb') as f:
            columnProfile = pickle.load(f)
        # read gt
        f = open(path + filename + ".txt", "r")
        gt = [line[:-1].split(' ') for line in f.readlines()]
        x_train.extend([ContionalRandomField.line2features(line.words) for line in columnProfile.lines])
        y_train.extend(gt)

    # # shuffle training data
    # tmp = list(zip(x_train, y_train))
    # random.shuffle(tmp)
    # x_train, y_train = zip(*tmp)

    # prepare the test data
    with open(path + testfile + '.pkl', 'rb') as f:
        columnProfile = pickle.load(f)
    f = open(path + testfile + ".txt", "r")
    gt = [line[:-1].split(' ') for line in f.readlines()]
    x_test = [ContionalRandomField.line2features(line.words) for line in columnProfile.lines]
    y_test = gt

    # X_train = [line2features(line.words) for line in columnProfile.lines]
    # y_train = gt

    model = CRF(
        algorithm='lbfgs',
        max_iterations=1000,
        all_possible_transitions=True
    )
    model.fit(x_train, y_train)

    # crf = ContionalRandomField(model)
    # # crf.predict(columnProfile)
    # with open('crf.pkl', 'wb') as f:
    #     pickle.dump(crf, f)

    y_pred = model.predict(x_test)
    # print(metrics.flat_accuracy_score(y_test, y_pred))

    TEST.extend(y_test)
    PRED.extend(y_pred)

print(metrics.flat_accuracy_score(TEST, PRED))

# test = [T for T in TEST]
test = [t == '1' for T in TEST for t in T]
pred = [t == '1' for T in PRED for t in T]
precision = average_precision_score(test, pred)
recall = recall_score(test, pred)
print(precision)
print(recall)

# columnProfile.drawLabel(y_pred)