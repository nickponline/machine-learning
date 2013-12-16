import sys
import pickle
import numpy as np
import scipy.sparse

from sklearn import svm, cross_validation, datasets
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import *
from sklearn.datasets import make_classification

def load(data, labs):
	# Load data in sparse format

	converters = {
		0 : lambda x: 1 if x == "HIGH" else 0,
	}

	M = np.genfromtxt(data, delimiter=',')
	y = np.genfromtxt(labs, delimiter=',', converters=converters)

	rows = M[0, :] - 1
	cols = M[1, :] - 1
	data = M[2, :]

	X = scipy.sparse.csr_matrix( (data,(rows,cols)) )
	return X, y

if __name__ == '__main__':

	# dataset to use: ip goip gobp etc ...	
	dataset = sys.argv[1]
	
	data   = '{}.txt'.format(dataset)
	labs   = '{}.lab'.format(dataset)
	model  = '{}.model'.format(dataset)
	result = '{}.results'.format(dataset)
	
	X, y = load(data, labs)
	S = StandardScaler(with_mean=False).fit(X)
	X = S.transform(X)
	X_orig = X.copy()
	y_orig = y.copy()

	est = None

	try:
		# and load train model from file.
		with open(model, mode='r') as fd:
			est = pickle.load(fd)

	except IOError:
		# otherwise retrain it.
		X, y = shuffle(X_orig, y_orig, random_state=0)
		kfold = cross_validation.StratifiedKFold (y, n_folds=10)
		grid  = GridSearchCV(cv=kfold, estimator=svm.SVC(kernel='linear', probability=True), param_grid=dict(C=np.logspace(-5, 5, 40)), n_jobs=-1, verbose=5)
		grid.fit(X, y)
		est = grid.best_estimator_

		# and save it.
		with open(model, mode='w') as fd:
			pickle.dump(est, fd)
	
	with open(result, mode='w') as fd:
		
		fd.write("accuracy,roc_auc,f1_score,matthews_corrcoef,precision,recall\n")
		
		# 50 runs of
		for i in xrange(50):
			X, y = shuffle(X_orig, y_orig, random_state=i)
			
			# 10 fold cross-validation
			kfold = cross_validation.StratifiedKFold (y, n_folds=10)

			acc = cross_validation.cross_val_score(est, X, y, cv=kfold, scoring = 'accuracy')
			auc = cross_validation.cross_val_score(est, X, y, cv=kfold, scoring = 'roc_auc')
			f1s = cross_validation.cross_val_score(est, X, y, cv=kfold, scoring = 'f1')
			pre = cross_validation.cross_val_score(est, X, y, cv=kfold, scoring = 'precision')
			rec = cross_validation.cross_val_score(est, X, y, cv=kfold, scoring = 'recall')
			mcc = cross_validation.cross_val_score(est, X, y, cv=kfold, scoring = make_scorer(lambda x,y : matthews_corrcoef(x, y), greater_is_better=True))
			
			ln = ''
			ln += ('%3.3f+/-%3.3f,'  % (acc.mean(), acc.std()))
			ln += ('%3.3f+/-%3.3f,'  % (auc.mean(), auc.std()))
			ln += ('%3.3f+/-%3.3f,'  % (f1s.mean(), f1s.std()))
			ln += ('%3.3f+/-%3.3f,'  % (pre.mean(), pre.std()))
			ln += ('%3.3f+/-%3.3f,'  % (rec.mean(), rec.std()))
			ln += ('%3.3f+/-%3.3f,'  % (mcc.mean(), mcc.std()))
			print ln
			fd.write(ln + '\n')
			
		# compute roc curve	
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10, random_state=0) 
		probas_ = est.fit(X_train, y_train).predict_proba(X_test)
		fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
		fd.write(",".join(map(str, fpr)) + '\n')
		fd.write(",".join(map(str, tpr)) + '\n')

