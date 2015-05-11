from __future__ import division

"""character_classifier.py: Character detection and classification in binary images."""

_author__ = "Philipp Hunziker"
__license__ = "GNU v.2"
__maintainer__ = "Philipp Hunziker"
__email__ = "hunzikp[at]gmail.com"
__status__ = "Development"

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
import scipy.ndimage as ndimage
import cPickle as pickle
from copy import copy
import json
import Tkinter as tk
import Image
import ImageTk
import bz2


class Component:
    def __init__(self, coords, im, connected_component_id, label=None, char_proba=None):
        self.coords = coords
        self.im = im
        self.label = label
        self.char_proba = char_proba
        self.connected_component_id = connected_component_id
        self.features = None

    def makefeatures(self, featurefactory):
        self.features = featurefactory.getfeatures(self.im)


class ImageComponents:

    def __init__(self):
        self.image = None
        self.component_image = None
        self.componentlist = []
        self.missinglist = []

    def __len__(self):
        return len(self.componentlist)

    def __addcomponent(self, component):
        if component.label is None:
            self.missinglist.append(True)
        else:
            self.missinglist.append(False)
            # Ensure label is string
            component.label = str(component.label)
        self.componentlist.append(component)

    def addimagecomponents(self, im, label=None, multi=True):
        self.image = im
        labeled_array, num_features = ndimage.measurements.label(im)
        self.component_image = labeled_array
        if multi:
            for f in range(1, num_features+1):
                yx = np.nonzero(labeled_array == f)
                if len(yx[0]) < 6:
                    continue
                xmin = np.min(yx[1])
                xmax = np.max(yx[1])
                ymin = np.min(yx[0])
                ymax = np.max(yx[0])
                coords = (xmin, xmax, ymin, ymax)
                this_im = copy(labeled_array[ymin:(ymax+1), xmin:(xmax+1)])
                this_im[this_im!=f] = 0
                this_im[this_im > 0] = 255
                this_im = this_im.astype('uint8', copy=True)
                component = Component(coords, this_im, f, label)
                self.__addcomponent(component)
        else:  # Only store largest connected component if multi option is turned off
            areas = []
            for f in range(1, num_features+1):
                area = np.sum(labeled_array==f)
                areas.append(area)
            largest_feature = (np.where(areas==max(areas))[0] + 1)[0]
            yx = np.nonzero(labeled_array == largest_feature)
            if len(yx[0]) >= 5:
                xmin = np.min(yx[1])
                xmax = np.max(yx[1])
                ymin = np.min(yx[0])
                ymax = np.max(yx[0])
                coords = (xmin, xmax, ymin, ymax)
                this_im = copy(labeled_array[ymin:(ymax+1), xmin:(xmax+1)])
                this_im[this_im!=f] = 0
                this_im[this_im > 0] = 255
                this_im = this_im.astype('uint8', copy=True)
                component = Component(coords, this_im, largest_feature, label=label)
                self.__addcomponent(component)

    def getlocationimage(self, index, zoom=(0.2, 0.5), width=300):
        component = self.componentlist[index]
        host_im = self.image
        coords = component.coords
        cxmin, cxmax, cymin, cymax = coords
        p1 = (cxmin, cymin)
        p2 = (cxmax, cymax)
        im_col = cv2.cvtColor(host_im, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(im_col, p1, p2, color=(0,0,255))
        cwidth = cxmax - cxmin + 1
        cheight = cymax - cymin + 1
        cwidth_add = round((cwidth/zoom[0] - cwidth)/2)
        cheight_add = round((cheight/zoom[1] - cheight)/2)
        zxmin = max(0, cxmin-cwidth_add)
        zxmax = min(host_im.shape[1], cxmax+cwidth_add)
        zymin = max(0, cymin-cheight_add)
        zymax = min(host_im.shape[0], cymax+cheight_add)
        im_zoom = im_col[zymin:zymax, zxmin:zxmax, :]
        zwidth = im_zoom.shape[1]
        zscale = width/zwidth
        im_zoom = cv2.resize(im_zoom, (0, 0), fx=zscale, fy=zscale, interpolation=cv2.INTER_CUBIC)
        return im_zoom

    def labeledimage(self, fontscale=0.8, thickness=1):
        outimage = copy(self.image)
        outimage[outimage>0] = 30
        nonmissing = self.getnonmissingids()
        for i in nonmissing:
            c = self.componentlist[i]
            label = c.label
            lowerleft = (c.coords[0], c.coords[3])
            fontface = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img=outimage, text=label, org=lowerleft, fontFace=fontface, fontScale=fontscale, color=255, thickness=thickness)
        return outimage

    def setlabel(self, index, label):
        if label is None:
            self.missinglist[index] = True
            self.componentlist[index].label = None
        else:
            self.missinglist[index] = False
            # Ensure label is string
            label = str(label)
            self.componentlist[index].label = label

    def getlabel(self, index):
        label = self.componentlist[index].label
        if label is None:
            label = ""
        return label

    def set_char_proba(self, index, char_proba):
        self.componentlist[index].char_proba = char_proba

    def get_char_proba(self, index):
        return self.componentlist[index].char_proba

    def getmissingcount(self):
        return sum(self.missinglist)

    def getmissingids(self):
        return [i for i, elem in enumerate(self.missinglist, 0) if elem]

    def getnonmissingids(self):
        return [i for i, elem in enumerate(self.missinglist, 0) if not elem]

    def ismissing(self, index):
        return self.missinglist[index]

    def makefeatures(self, featurefactory):
        for c in self.componentlist:
            c.makefeatures(featurefactory)

    def gettrainingdata(self, asnp=False):
        nonmissing = self.getnonmissingids()
        features = []
        labels = []
        for i in nonmissing:
            component = self.componentlist[i]
            features.append(component.features)
            label_str = component.label
            labels.append(label_str)
        if asnp:
            return np.asarray(features), np.asarray(labels)
        else:
            return features, labels

    def getfeaturedata(self, asnp=False):
        select = range(0, len(self.componentlist))
        features = []
        for i in select:
            component = self.componentlist[i]
            features.append(component.features)
        if asnp:
            return np.asarray(features)
        else:
            return features

    def removecomponent(self, index):
        c = self.componentlist[index]
        self.componnentlist = self.componentlist[0:index] + self.componentlist[index:len(self.componentlist)]
        self.missinglist = self.missinglist[0:index] + self.missinglist[index:len(self.missinglist)]
        connected_component_id = c.connected_component_id
        self.image[self.component_image==connected_component_id] = 0
        self.component_image[self.component_image==connected_component_id] = 0

    def removecomponents(self, bool_list):
        for i in range(0, len(bool_list)):
            if bool_list[i]:
                self.removecomponent(i)


class ImageComponentCollection:

    def __init__(self):
        self.iclist = []
        self.iclengths = []

    def addimagecomponents(self, ic):
        self.iclist.append(ic)
        self.iclengths.append(len(ic))

    def addimagecomponentlist(self, icl):
        for ic in icl:
            self.addimagecomponents(ic)

    def setlabel(self, imageid, componentid, label):
        self.iclist[imageid].setlabel(componentid, label)

    def getlabel(self, imageid, componentid):
        return self.iclist[imageid].getlabel(componentid)

    def getmissingcount(self):
        mc = 0
        for ic in self.iclist:
            mc += ic.getmissingcount()
        return mc

    def makefeatures(self, featurefactory):
        for ic in self.iclist:
            ic.makefeatures(featurefactory)

    def gettrainingdata(self):
        features, labels = self.iclist[0].gettrainingdata()
        if len(self.iclist) > 1:
            for i in range(1, len(self.iclist)):
                f, l = self.iclist[i].gettrainingdata()
                features = features + f
                labels = labels + l
        feature_np = np.asarray(features)
        label_np = np.asarray(labels)
        return feature_np, label_np

    def getfeaturedata(self):
        features = self.iclist[0].getfeaturedata()
        if len(self.iclist) > 1:
            for i in range(1, len(self.iclist)):
                f = self.iclist[i].getfeaturedata()
                features + features + f
        feature_np = np.asarray(features)
        return feature_np

    def getnextcomponentid(self, image_id, component_id, step):
        if step > 0:
            thislength = self.iclengths[image_id]
            if component_id == thislength - 1:
                next_component_id = 0
                if image_id == len(self.iclist) - 1:
                    next_image_id = 0
                else:
                    next_image_id = image_id + 1
            else:
                next_image_id = image_id
                next_component_id = component_id + 1
        else:
            if component_id == 0:
                if image_id == 0:
                    next_image_id = len(self.iclist) - 1
                else:
                    next_image_id = image_id - 1
                next_component_id = self.iclengths[next_image_id] - 1
            else:
                next_image_id = image_id
                next_component_id = component_id - 1
        return [next_image_id, next_component_id]

    def getlocationimage(self, image_id, component_id):
        return self.iclist[image_id].getlocationimage(component_id)

    def exportComponents(self, filename):
        file = self.iclist
        with open(filename, 'wb') as output:
            pickle.dump(file, output, pickle.HIGHEST_PROTOCOL)

    def importComponents(self, filename):
        iclist = pickle.load(open( filename, "rb"))
        self.addimagecomponentlist(iclist)

    def ismissing(self, image_id, component_id):
        return self.iclist[image_id].ismissing(component_id)


class ClassificationGUI:

    def __init__(self, icc):
        self.icc = icc
        self.image_id = 0
        self.component_id = 0
        self.root = tk.Tk()

        # ID Display
        self.idvar = tk.StringVar()
        self.idlabel = tk.Label(self.root, textvariable=self.idvar)
        self.idlabel.pack()

        # Missing display
        self.missingv = tk.StringVar()
        self.missinglabel = tk.Label(self.root, textvariable=self.missingv, fg="red")
        self.missinglabel.pack()

        # Missing count
        self.mv = tk.StringVar()
        self.mlabel = tk.Label(self.root, textvariable=self.mv)
        self.mlabel.pack()

        # Only missing option
        self.mov = tk.IntVar()
        self.missingonly = False
        self.missing_checkbutton = tk.Checkbutton(self.root, text="Missing Only", variable=self.mov, command=self.missing_checkbutton_callback)
        self.missing_checkbutton.pack()

       # Get and display image
        cim = np.zeros((100,100))
        cim.astype("uint8")
        img = Image.fromarray(cim)
        tkim = ImageTk.PhotoImage(img)
        self.panel = tk.Label(self.root, image = tkim)
        self.panel.pack(fill="both", expand="yes")
        self.panel.tkim = tkim

        # Entry
        self.e = tk.Entry(self.root)
        self.e.pack()
        self.e.focus_set()

        # Buttons
        self.next_button = tk.Button(self.root, text="Next", width=10, command=self.next_button_callback)
        self.next_button.pack(side="left")
        self.prev_button = tk.Button(self.root, text="Previous", width=10, command=self.prev_button_callback)
        self.prev_button.pack(side="left")
        self.skip_button = tk.Button(self.root, text="Skip", width=10, command=self.skip_button_callback)
        self.skip_button.pack(side="left")
        self.save_button = tk.Button(self.root, text="Save", width=10, command=self.save_button_callback)
        self.save_button.pack()

        # Key bindings
        self.e.bind("<Return>", self.enter_key_callback)
        self.e.bind("<Left>", self.left_key_callback)
        self.e.bind("<Right>", self.right_key_callback)
        self.e.bind("<Escape>", self.esc_key_callback)

    def update_display(self):
        next_cim = self.icc.getlocationimage(self.image_id, self.component_id)
        next_img = Image.fromarray(next_cim)
        next_tkim = ImageTk.PhotoImage(next_img)
        self.panel.configure(image = next_tkim)
        self.panel.tkim = next_tkim
        self.e.delete(0, tk.END)
        self.e.insert(0, self.icc.getlabel(self.image_id, self.component_id))
        self.mv.set("Total Missing: " + str(self.icc.getmissingcount()))
        self.missingv.set(self.missingstring())
        self.idvar.set(str(self.image_id) + ", " + str(self.component_id))

    def next_button_callback(self):
        self.icc.setlabel(self.image_id, self.component_id, self.e.get())
        self.image_id, self.component_id = self.icc.getnextcomponentid(self.image_id, self.component_id, 1)
        self.update_display()

    def prev_button_callback(self):
        self.image_id, self.component_id = self.icc.getnextcomponentid(self.image_id, self.component_id, -1)
        self.update_display()

    def skip_button_callback(self):
        self.image_id, self.component_id = self.icc.getnextcomponentid(self.image_id, self.component_id, 1)
        self.update_display()

    def missing_checkbutton_callback(self):
        self.missingonly = self.mov.get() == 1

    def save_button_callback(self):
        self.icc.exportComponents(filename="data/chartraining/handtrained/export.pkl")
        print "Saved!"

    def enter_key_callback(self, event):
        self.next_button_callback()

    def left_key_callback(self, event):
        self.prev_button_callback()

    def right_key_callback(self, event):
        self.skip_button_callback()

    def esc_key_callback(self, event):
        self.root.destroy()
        self.root = None

    def launch(self):
        if self.root is None:
            self.__init__(self.icc)
        self.update_display()
        self.root.mainloop()

    def missingstring(self):
        if self.icc.ismissing(self.image_id, self.component_id):
            return "Missing"
        else:
            return "Classified"


class FeatureFactory:

    def __init__(self, size):
        self.size = size

    def getfeatures(self, im):
        im_standard = self.resize(self.addBorder(self.crop(im)), self.size)
        pixelfeatures = im_standard.flatten().tolist()
        return pixelfeatures

    def crop(self, iim):
        im = copy(iim)
        xy = np.nonzero(im > 0)
        if len(xy[0]) > 0:
            xmin = np.min(xy[0])
            xmax = np.max(xy[0])
            ymin = np.min(xy[1])
            ymax = np.max(xy[1])
            im = im[xmin:(xmax+1), ymin:(ymax+1)]
        return im

    def addBorder(self, iim):
        in_im = copy(iim)
        im_dim = in_im.shape
        height = im_dim[0]
        width = im_dim[1]
        if height > width:
            res = height - width
            add = int(round(res/2, 0))
            im_border = cv2.copyMakeBorder(in_im, 0, 0, add, add, cv2.BORDER_CONSTANT, value=[0])
        else:
            res = width - height
            add = int(round(res/2, 0))
            im_border = cv2.copyMakeBorder(in_im, add, add, 0, 0, cv2.BORDER_CONSTANT, value=[0])
        return im_border

    def resize(self, iim, size):
        im_resize = cv2.resize(iim, (size, size), interpolation=cv2.INTER_CUBIC)
        im_resize[im_resize > 0] = 255
        return im_resize


class CharacterClassifier:

    def __init__(self, icc=None, skcl=None, featurefactory=None):
        if skcl is None:
            self.skcl = RandomForestClassifier()
        else:
            self.skcl = None
        if icc is None:
            self.icc = ImageComponentCollection()
        else:
            self.icc = icc
        if featurefactory is None:
            self.featurefactory = FeatureFactory(size=30)
        else:
            self.featurefactory = featurefactory
        self.labeldict = {}

    def set_image_component_collection(self, icc):
        self.icc = icc

    def set_feature_factory(self, featurefactory):
        self.featurefactory = featurefactory

    def label2factor(self, labels):
        ulabels = np.unique(labels)
        self.labeldict = {}
        for i in range(0, len(ulabels)):
            self.labeldict[ulabels[i]] = i
        factors = []
        for i in range(0, len(labels)):
            factors.append(self.labeldict[labels[i]])
        factors = np.array(factors)
        return factors

    def factor2label(self, factors):
        labels = []
        for i in range(0, len(factors)):
            labels.append(self.labeldict.keys()[self.labeldict.values().index(factors[i])])
        labels = np.array(labels)
        return labels

    def tune(self, par_list, cv):
        self.icc.makefeatures(self.featurefactory)
        X, y = self.icc.gettrainingdata()
        factors = self.label2factor(y)
        par_list = np.asarray(par_list)
        score_vec = []
        for par in par_list:
            clf_rf = self.skcl
            clf_rf.n_estimators = par
            scores = cross_validation.cross_val_score(clf_rf, X, factors, cv=cv, scoring='accuracy')
            score_vec.append(scores.mean())
            print "par: " + str(par) + ", score:" + str(scores.mean())
        par_opt = par_list[score_vec == max(score_vec)][0]
        self.skcl.n_estimators = par_opt
        self.skcl.fit(X, factors)

    def predict_image_components(self, ic, predict_proba=False):
        # Make features from input ic
        ic = copy(ic)
        ic.makefeatures(self.featurefactory)
        X = ic.getfeaturedata()
        # Predict
        factors = self.skcl.predict(X)
        labels = self.factor2label(factors)
        for i in range(0, len(ic)):
            ic.setlabel(i, labels[i])
        if predict_proba:
            char_probs = self.skcl.predict_proba(X)
            anychar_probs = 1 - char_probs[:, 0]
            for i in range(0, len(ic)):
                ic.setproba(i, anychar_probs[i])
        return ic

    def predict_image(self, im, predict_proba=False):
        # Make features from input image
        ic = ImageComponents()
        ic.addimagecomponents(copy(im), multi=True)
        ic = self.predict_image_components(ic, predict_proba)
        return ic

    def save(self, filename):
        file = [self.skcl, self.featurefactory]
        with bz2.BZ2File(filename, 'w') as output:
            pickle.dump(file, output, pickle.HIGHEST_PROTOCOL)
        filename_json = filename.replace("pkl", "json")
        with open(filename_json, 'w') as f:
            f.write(json.dumps(self.labeldict))

    def load(self, filename):
        with bz2.BZ2File(filename, 'rb') as f:
            skcl, featurefactory = pickle.load(f)
        self.skcl = skcl
        self.featurefactory = featurefactory
        filename_json = filename.replace("pkl", "json")
        self.labeldict = json.load(open( filename_json))
