'''
A script that takes in test data and runs a TBL classifier on it, and prints the transformations and the results
Command to run: TBL_classify.py test_data model_file output_filename max_trans
where
- test_data is the file with the test data
- model_file is name of the model_file that contains a TBL model to read in
- sys_output is the output file that records the steps taken by the TBL classifier, in form:
instanceName trueLabel sysLabel transformation1 transformation2....
where transformations contain 'feature from_label to_label'
- max_trans is the maximum transformations in the model file to use
'''

import argparse
from helpful_functions import svm_light_to_binary_TDL_features, calc_accuracy
from TBL_train import apply_trans_to_data


def read_TBL_model(model_filename, max_trans):
    default, transitions, gains = None, [], []
    trans_read = 0
    with open(model_filename, 'r') as mfile:
        for line in mfile:
            if trans_read >= max_trans:
                break
            if line:
                line = line.strip().split()
                if len(line) == 1:
                    default = line[0]
                else:
                    transitions.append(tuple(line[:3]))
                    gains.append(float(line[3]))
                    trans_read += 1
    return default, transitions, gains


def classify_TBL(test_records, TBL_model):
    # will have to track all the transformations that happen
    trans_applied = [[] for i in range(len(test_records))]
    gold_labels, final_predicted_labels = [], []
    for trans in TBL_model:
        apply_trans_to_data(test_records, trans, True, trans_applied)
    # grab gold and predicted labels for accuracy calcs
    for gold, pred, feats in test_records:
        gold_labels.append(gold)
        final_predicted_labels.append(pred)

    return trans_applied, gold_labels, final_predicted_labels


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('test_filename', help='the filename with the test data, svm-light format')
    p.add_argument('model_filename', help='the filename with the model to use for TBL classification')
    p.add_argument('output_filename', help='the filename with to write the sys_out from TBL classification')
    p.add_argument('max_trans', type=int, help='the maximum transitions to use in each model file')
    args = p.parse_args()

    # read model
    default_label, TBL_model, model_gains = read_TBL_model(args.model_filename, args.max_trans)
    # read test data
    test_records, all_class_list, all_feat_list = svm_light_to_binary_TDL_features(args.test_filename, default_label)
    # TBL classify
    applied_trans, golds, preds = classify_TBL(test_records, TBL_model)
    # write output
    with open(args.output_filename, 'w') as outfile:
        for i in range(len(applied_trans)):
            if applied_trans[i]:
                curr_trans = ' '.join([' '.join(applied_trans[i][j]) for j in range(len(applied_trans[i]))])
            else:
                curr_trans = '*no transformations applied*'
            outfile.write('file{} {} {} {}\n'.format(i, golds[i], preds[i], curr_trans))
    # calc baseline accuracy (with initial annotator)
    baseline = [default_label for i in range(len(golds))]
    calc_accuracy(golds, baseline, all_class_list, 'baseline')
    # cal TBL acc
    calc_accuracy(golds, preds, all_class_list, 'test')