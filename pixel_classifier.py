from __future__ import division

"""pixel_classifier.py: Supervised image pixel classification using random forests."""

_author__ = "Philipp Hunziker"
__license__ = "GNU v.2"
__maintainer__ = "Philipp Hunziker"
__email__ = "hunzikp[at]gmail.com"
__status__ = "Development"

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from itertools import product
import cv2
import numpy as np
import cPickle as pickle


class PixelClassifier:

    def __init__(self):
        self.clf = None
        self.im_class = None
        self.im_orig = None
        self.lothres = None
        self.upthres = None
        self.im_class_thres = None
        self.filtername = None
        self.filter_d = None
        self.filter_sigmacolor = None
        self.filter_sigmaspace = None

    def trained(self):
        if self.clf is not None:
            return True
        else:
            return False

    def set_training_images(self, im_class, im_orig, lothres=np.array([0, 254, 0], dtype=np.uint8), upthres=np.array([1, 255, 1], dtype=np.uint8)):
        self.im_class = im_class
        self.im_orig = im_orig
        self.lothres = lothres
        self.upthres = upthres
        self.im_class_thres = cv2.inRange(self.im_class, self.lothres, self.upthres)
        self.im_class_thres[self.im_class_thres > 0] = 255

    def get_training_data(self, filtername, filter_d, filter_sigmacolor, filter_sigmaspace):
        if filtername == "bilateral":
            im_filter = cv2.bilateralFilter(self.im_orig, filter_d, filter_sigmacolor, filter_sigmaspace)
        else:
            raise ValueError("Only bilateral filtering is allowed.")

        train_data = []
        for i in range(0, im_filter.shape[0]):
            for j in range(0, im_filter.shape[1]):
                colors = tuple(im_filter[i,j])
                # Remove purely white training data
                if colors[0] <= 255 or colors[1] <= 255 or colors[2] <= 255:
                    label = (self.im_class_thres[i,j],)
                    coords = (i,j)
                    train_data.append(coords + colors + label)
        train_data = np.asarray(train_data)
        return train_data

    def fit(self, n_estimators=10, filtername="bilateral", filter_d=3, filter_sigmacolor=1, filter_sigmaspace=1):
        self.filtername = filtername
        self.filter_d = filter_d
        self.filter_sigmacolor = filter_sigmacolor
        self.filter_sigmaspace = filter_sigmaspace
        train_data = self.get_training_data(filtername=filtername, filter_d=filter_d, filter_sigmacolor=filter_sigmacolor, filter_sigmaspace=filter_sigmaspace)
        rf = RandomForestClassifier(n_estimators=n_estimators)
        rf.fit(train_data[:,2:5], train_data[:,5])
        self.clf = rf

    def get_cv_score(self, n_estimators, filtername, filter_d, filter_sigmacolor, filter_sigmaspace, cv):
        train_data = self.get_training_data(filtername=filtername, filter_d=filter_d, filter_sigmacolor=filter_sigmacolor, filter_sigmaspace=filter_sigmaspace)
        rf = RandomForestClassifier(n_estimators=n_estimators)
        scores = cross_validation.cross_val_score(rf, train_data[:,2:5], train_data[:,5], cv=cv)
        return scores.mean()

    def tune(self, n_estimators_list=[10], filtername_list=["bilateral"], filter_d_list=[3], filter_sigmacolor_list=[1], filter_sigmaspace_list=[1], cv=3):
        params = []
        maxi = 0
        maxscore = -1
        i = 0
        for filtername in filtername_list:
            if filtername == "bilateral":
                for n_estimators, filter_d, filter_sigmacolor, filter_sigmaspace in product(n_estimators_list, filter_d_list, filter_sigmacolor_list, filter_sigmaspace_list):
                    params.append([n_estimators, filtername, filter_d, filter_sigmacolor, filter_sigmaspace])
                    score = self.get_cv_score(n_estimators, filtername, filter_d, filter_sigmacolor, filter_sigmaspace, cv=cv)
                    if score > maxscore:
                        maxi = i
                        maxscore = score
                    print "tested " + str(params[i]) + ", score: " + str(score)
                    i += 1
            else:
                raise ValueError("Only bilateral filtering is allowed.")

        best_params = params[maxi]
        self.fit(n_estimators=best_params[0], filtername=best_params[1], filter_d=best_params[2],
                 filter_sigmacolor=best_params[3], filter_sigmaspace=best_params[4])

    def predict(self, im):
        if self.filtername == "bilateral":
            im_filter = cv2.bilateralFilter(im, self.filter_d, self.filter_sigmacolor, self.filter_sigmaspace)
        else:
            raise ValueError("Only bilateral filtering is allowed.")
        pred_data = []
        for i in range(0, im_filter.shape[0]):
            for j in range(0, im_filter.shape[1]):
                colors = tuple(im_filter[i,j])
                coords = (i,j)
                pred_data.append(coords + colors)
        pred_data = np.asarray(pred_data)
        classified = self.clf.predict(pred_data[:,2:5])

        lower_none= np.array([255, 255, 255], dtype=np.uint8)
        upper_none= np.array([255, 255, 255], dtype=np.uint8)
        im_classified = cv2.inRange(im, lower_none, upper_none)
        im_classified[im_classified > 0] = 0
        for l in range(0, len(pred_data)):
            r = pred_data[l,:]
            i = r[0]
            j = r[1]
            pval = classified[l]
            im_classified[i,j] = pval
        return im_classified

    def save(self, path):
        with open(path, 'wb') as output:
            pickle.dump([self.clf, self.filtername, self.filter_d, self.filter_sigmacolor, self.filter_sigmaspace],
                        output, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        self.clf, self.filtername, self.filter_d, \
            self.filter_sigmacolor, self.filter_sigmaspace = pickle.load(open(path, "rb"))