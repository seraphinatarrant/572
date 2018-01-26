from sklearn.metrics import confusion_matrix

def calc_accuracy(gold_standard, predictions, classes, type):
    '''
    :param gold_standard: array of gold standard values
    :param predictions: array of predicted values
    :param classes: a list of class labels for labeling the columns and rows of the matric
    :param type: training or test
    :return: nothing - just prints
    '''
    header = 'Confusion matrix for the {} data:\nrow is the truth, column is the system output\n'.format(type.lower())
    confusion = confusion_matrix(gold_standard, predictions, labels=classes)
    correct, total = confusion.diagonal().sum(), confusion.sum()
    accuracy = correct/total
    max_char = max(map(len, classes))
    #print accuracy data
    print(header)
    print('{0:{1}} {2}'.format(' ', max_char, ' '.join(classes)))
    for index in range(len(classes)):
        print('{0:{1}} {2}'.format(classes[index], max_char+5, ' '.join([str(num) for num in confusion[index]])))
    print('\n{} accuracy={:.5f}\n\n'.format(type, accuracy))

if __name__ == "__main__":
    print('Hello I am only a functions helper file! Love me anyway!')
    