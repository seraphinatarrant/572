'''
A script that calculates the MaxEnt model expectation for given svm-light training data. If a model file is given, the
weights from that file are used, else the Maximum Entropy probabilities are assumed.
Note that this is mostly intended to be readable - since it is not heavy lifting. So there are fancier ways of doing it
via numpy that I didn't do, since this is intuitive and readable.

Command to run: calc_model_exp.py training_data output_file {model_file} where model_file is optional
'''

import argparse
from collections import defaultdict, Counter

from maxent_classify import svm_light_to_binary_features, read_maxent_model
from calc_emp_exp import write_expec


def calc_model_exp(train_records, model_weights=None):
    '''
    Calculates model expectation given training records, and possibly a MaxEnt model. Note that it does not divide by
    number of records - that is done at the print step.
    :param train_records: a nested list [[gold label, set of features],...]
    :param model_weights: if it exists, it's a nested dict of class: feature: weight from a MaxEnt model
    :return: a nested dict Counter of form {class_label: {feature: count}}
    '''
    model_exp_dict = defaultdict(Counter)
    if model_weights:
        all_classes = list(model_weights)
    else:
        all_classes = set([rec[0] for rec in train_records])
    num_classes = len(all_classes)
    for record in train_records:
        label, features = record
        # since this is the model, we pretend we do not know the class label and iterate through all of them
        for class_label in all_classes:
            for feat in features:
                # if a model was given use that, else the weight will be 1/num_classes (ie maximum entropy)
                if model_weights:
                    model_exp_dict[class_label][feat] += model_weights[class_label][feat]
                else:
                    model_exp_dict[class_label][feat] += 1/num_classes

    return model_exp_dict



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('train_filename', help='A training file in svm-light format')
    p.add_argument('output_filename', help='A filename to write the empirical expectations to')
    p.add_argument('-m', '--model_filename',
                   help='An opt filename w/ a MaxEnt model to use in model expectation calculation')
    args = p.parse_args()

    #check if model file exists, if so, read model
    if args.model_filename:
        c_weights, f_weights = read_maxent_model(args.model_filename)
    else:
        f_weights = None
    # if so, read model if not, initialise a class_weights dict where each
    train_records = svm_light_to_binary_features(args.train_filename)
    expectations = calc_model_exp(train_records, f_weights)
    write_expec(expectations, len(train_records), args.output_filename, 'model')

    