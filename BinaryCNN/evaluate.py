
# create labels, 1 for ME, 0 for text
pred = open('pred.txt').readlines()
gt = open('tgt-test.txt').readlines()

if len(pred) != len(gt):
    raise Exception('length is not equal')

cnt = 0
tp = fn = fp = tn = 0
for i in range(len(pred)):
    # y = pred[i]
    y = '1' if float(pred[i]) > 0.5 else '0'
    if y == gt[i][0]:
        cnt += 1
    if y == '1' and gt[i][0] == '1':
        tp += 1
    if y == '0' and gt[i][0] == '1':
        fn += 1
    if y == '1' and gt[i][0] == '0':
        fp += 1
    if y == '0' and gt[i][0] == '0':
        tn += 1

accuracy = cnt / len(pred)
print(accuracy)