from sklearn.metrics import confusion_matrix
import numpy as np

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

def cosine_sim(vec, vec_array):
    '''
    :param vec: a vector
    :param vec_array: an array of vectors
    :return: cosine_sim_array of the cosine similarity between the vector and each vector in the array
    '''
    #take dot product of 2 vectors. which reduces dimensionality and gives me an array of results.
    #IMPORTANT that vec_array is first arg as a result
    dot_prod_array = np.dot(vec_array, vec)
    len_vec_array, len_x_d = (vec_array**2).sum(axis=1) ** .5, (vec ** 2).sum() ** .5
    cosine_sim_array = np.divide(dot_prod_array, len_vec_array*len_x_d)
    #best_vec_index = np.argmax(cosine_sim_array)

    return cosine_sim_array


def euclidean_dist(vec, vec_array):
    '''
    Compares given vector with array of vectors and finds one with closest Euclidean distance.
    :param vec: a vector
    :param vec_array: array of vectors
    :return: an array of the euclidean distance results
    '''
    #Sqrt of (sum of (x_i - y_i)^2 for all n of vector)
    results = (((vec - vec_array) ** 2).sum(axis=1)) ** .5 #axis = 0 sums over columns, axis = 1 sums over rows
    #best_vec_index = np.argmin(results)
    return results

if __name__ == "__main__":
    print('Hello I am only a functions helper file! Love me anyway!')
    