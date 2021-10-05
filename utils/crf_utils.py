
# conditional random field class
class ContionalRandomField:
    def __init__(self, model):
        self.model = model

    def predict(self, imageProfile):
        x_test = [ContionalRandomField.line2features(line.words) for line in imageProfile.lines]
        y_pred = self.model.predict(x_test)
        # imageProfile.drawWords()
        self.udpateCRFlabels(imageProfile, y_pred)
        # imageProfile.drawWords()

    def udpateCRFlabels(self, imageProfile, labels):
        for i, line in enumerate(imageProfile.lines):
            for j, word in enumerate(line.words):
                word.me_label = False if labels[i][j] == '0' else True

    @staticmethod
    def word2features(line, i):
        word = line[i]
        features = {
            'me_likelihood': word.me_likelihood,
            'is_stopword': word.is_stopword,
            'width': word.width,
            'height': word.height,
            'text': word.text,
            'is_plaintext': word.is_plaintext,
        }
        if i > 0:
            word1 = line[i - 1]
            features.update({
                '-1:me_likelihood': word1.me_likelihood,
                '-1:is_stopword': word1.is_stopword,
                '-1:dist': word.left-word1.right,
                '-1:text': word1.text,
            })
        else:
            features['BOS'] = True
        if i < len(line) - 1:
            word2 = line[i + 1]
            features.update({
                '+1:me_likelihood': word2.me_likelihood,
                '+1:is_stopword': word2.is_stopword,
                '+1:dist': word2.left - word.right,
                '+1:text': word2.text,
            })
        else:
            features['EOS'] = True

        return features

    @staticmethod
    def line2features(line):
        return [ContionalRandomField.word2features(line, i) for i in range(len(line))]

