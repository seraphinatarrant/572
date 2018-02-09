'''
A script that calculates the MaxEnt empirical expectation given training data in svm-light format.
Command to run: calc_emp_exp.py training_data output_file
'''

import argparse
from collections import defaultdict, Counter

import sys
from maxent_classify import svm_light_to_binary_features


def calc_emp_exp(train_records):
    '''
    Calculates empirical expectation given training records. Note that it does not divide by number of records - that
    is done at the print step.
    :param train_records: a nested list [[gold label, set of features],...]
    :return: a nested dict Counter of form {class_label: {feature: count}}
    '''
    emp_exp_dict = defaultdict(Counter)
    for record in train_records:
        label, features = record
        for feat in features:
            emp_exp_dict[label][feat] += 1

    return emp_exp_dict


def write_expec(exp_dict, num_records, outputfilename, type):
    '''
    writes expectations to a file in form: 'class_label feature expectation raw_count'
    :param exp_dict: the nested dict Counter of form {class_label: {feature: count}}
    :param num_records: 'N' in maxent terminology - the total number of samples
    :param outputfilename: filename to write data to
    :param type: the type to write: empirical or model. Decides the formatting of raw count which will be float or int
    :return: None - writes to file
    '''
    output_data = []
    class_list = sorted(list(exp_dict))  # a sorted list of all classes
    for class_label in class_list:
        feature_list = sorted(list(exp_dict[class_label]))  # a sorted list of features. If I don't want to iterate over the whole vocab
        for feat in feature_list:
            val = exp_dict[class_label][feat]
            # if data is empirical, the val is a raw count and the expectation is val/num_records
            if type == 'empirical':
                output_data.append('{} {} {:.5f} {}'.format(class_label, feat, val / num_records, val))
            # if data is model expectation, the val is the expectation and the raw count is expectation * num_records (printed as int)
            elif type == 'model':
                output_data.append('{} {} {:.5f} {:.5f}'.format(class_label, feat, val / num_records, val))
            else:
                print('Invalid type of expectation given: please give either "empirical" or "model"', file=sys.stderr)

    with open(outputfilename, 'w') as outfile:
        outfile.write('\n'.join(output_data))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('train_filename', help='A training file in svm-light format')
    p.add_argument('output_filename', help='A filename to write the empirical expectations to')
    args = p.parse_args()
    train_records = svm_light_to_binary_features(args.train_filename)
    expectations = calc_emp_exp(train_records)
    write_expec(expectations, len(train_records), args.output_filename, 'empirical')
    