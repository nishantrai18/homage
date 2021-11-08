import time

from sklearn import metrics
from sklearn.linear_model import RidgeClassifier
from sklearn.cluster import MiniBatchKMeans


class QuickSupervisedModelTrainer(object):

    def __init__(self, num_classes, modes):
        self.modes = modes
        self.mode_pairs = [(m0, m1) for m0 in self.modes for m1 in self.modes if m0 < m1]
        self.ridge = {m: RidgeClassifier() for m in self.modes}
        self.kmeans = {m: MiniBatchKMeans(n_clusters=num_classes, random_state=0, batch_size=256) for m in self.modes}

    def evaluate_classification(self, trainD, valD):
        tic = time.time()
        trainY, valY = trainD["Y"].cpu().numpy(), valD["Y"].cpu().numpy()
        print(trainY.shape, valY.shape)
        for mode in self.modes:
            trainX, valX = trainD["X"][mode].cpu().numpy(), valD["X"][mode].cpu().numpy()
            print(trainX.shape, valX.shape)
            self.ridge[mode].fit(trainX, trainY)
            score = round(self.ridge[mode].score(valX, valY), 3)
            print("--- Mode: {} - RidgeAcc: {}".format(mode, score))
        print("Time taken to perform classification evaluation:", time.time() - tic)

    def fit_and_predict_clustering(self, data, tag):
        tic = time.time()
        preds = {}
        for mode in self.modes:
            preds[mode] = self.kmeans[mode].fit_predict(data["X"][mode].cpu().numpy())
        print("Time taken to perform {} clustering:".format(tag), time.time() - tic)
        return preds

    def evaluate_clustering_based_on_ground_truth(self, preds, label, tag):
        tic = time.time()
        return_dict = {}
        for mode in self.modes:
            ars = round(metrics.adjusted_rand_score(preds[mode], label), 3)
            v_measure = round(metrics.v_measure_score(preds[mode], label), 3)
            print("--- Mode: {} - Adj Rand. Score: {}, V-Measure: {}".format(mode, ars, v_measure))
            return_dict[mode] = dict(ars=ars, v_measure=v_measure)
        print("Time taken to evaluate {} clustering:".format(tag), time.time() - tic)
        return return_dict

    def evaluate_clustering_based_on_mutual_information(self, preds, tag):
        tic = time.time()
        return_dict = {}
        for m0, m1 in self.mode_pairs:
            ami = round(metrics.adjusted_mutual_info_score(preds[m0], preds[m1], average_method='max'), 3)
            v_measure = round(metrics.v_measure_score(preds[m0], preds[m1]), 3)
            print("--- Modes: {}/{} - Adj MI: {}, V Measure: {}".format(m0, m1, ami, v_measure))
            return_dict['{}_{}'.format(m0, m1)] = dict(ami=ami, v_measure=v_measure)
        print("Time taken to evaluate {} clustering MI:".format(tag), time.time() - tic)
        return return_dict

    def evaluate_clustering(self, data, tag):
        '''
            Need to evaluate clustering using the following methods,
            1. Correctness of clustering based on ground truth labels
                a. Adjusted Rand Score
                b. Homogeneity, completeness and V-measure
            2. Mutual information based scores (across modalities)
        '''
        label = data["Y"].cpu().numpy()

        preds = self.fit_and_predict_clustering(data, tag)
        return_dict = {}
        return_dict['gt'] = self.evaluate_clustering_based_on_ground_truth(preds, label, tag)
        return_dict['mi'] = self.evaluate_clustering_based_on_mutual_information(preds, tag)
        
        return return_dict
