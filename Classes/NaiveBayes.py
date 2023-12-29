import numpy as np

class ourNaiveBayes:
    def set_priors(self, X, y):
        n_rows = X.shape[0]
        classes, counts = np.unique(y, return_counts=True) # return how many samples there are of each class
        # prior probability of a sample being from a certain class
        self.priors = {class_: count / n_rows for class_, count in zip(classes, counts)}
    
    def set_bin_number(self, h):
        self.h = h

    def estimate_pmf_hist(self, X):
         # return number of samples in each bin and edge values of the bin
        counts, edges = np.histogram(X, bins=self.h)
        # list of probabilities of a feature being from a certain value range
        cond = []
        # total number of samples
        total = np.sum(counts)
        for upper_bound, c in zip(edges[1:], counts):
            # get percentage of samples in bin and append ending point of the bin
            # and percentage to the conditionals list
            if c != 0:
                curr_cond = c / total
            # if a bin is not assigned any points (i.e there were no features in that value range)
            # assign it a very very low probability rather than a flat 0
            # chose to do this because a single PIXEL not in a yet encountered value range for that
            # feature for the given class immediately disqualifies that picture being from that class
            # otherwise, so instead give it a big penalty without immediately disqualifying it
            else:
                curr_cond = 0.001
            cond.append((upper_bound.real, curr_cond))
        return cond
    
    def set_conditionals(self, X, y):
        N = X.shape[1] # number of features
        class_labels = np.unique(y)
        c = class_labels.shape[0] # number of classes
        # get conditional probabilities of features given the sample is from a certain class
        self.conditionals = {class_ : {} for class_ in class_labels}
        for class_ in range(c):
            # get features and labels corresponding to samples from the current class
            rows_subset = X[y == class_]
            for feature in range(N):            
                features_subset = rows_subset[:, feature] # get vector of values of a certain feature in all samples
                                                          # of the current class
                # get probabilities of feature value ranges given the current class
                estimate = self.estimate_pmf_hist(features_subset)
                # Note: this should never happen if we cleaned the data set, but just in case for debugging
                if estimate == 0 or estimate == None:
                    print(f"no bins at feature {feature} in class {class_}")
                self.conditionals[class_][feature] = estimate
        
    def predict_probabilities(self, X):
        N = X.shape[1] #number of features
        n_rows = X.shape[0]
        c = len(self.priors) #number of classes
        # matrix of class probabilities for each sample
        probs = np.zeros((n_rows, c))
        # for each sample
        for i in range(n_rows):
            # current sample
            x = X[i]
            # vector of class probabilities for the current sample
            class_probs = np.zeros((c, 1))
            # for each class
            for class_ in range(c):
                # get class prior
                prior = self.priors[class_]
                # initialize the probability of the sample belonging to that class
                # as the prior of that class
                class_prob = prior
                # for each feature
                for j in range(N):
                    feature = x[j]
                    # get probability of current feature
                    # if it doesn't exist in the conditionals dictionary return a very very small probability
                    curr_pmf = self.conditionals[class_][j]
                    feature_prob = 0.001
                    # find value interval in which the feature is and assign it
                    # the conditional probability corresponding to that interval
                    for upper_bound, cond_prob in curr_pmf:
                        if upper_bound < feature:
                            continue
                        else:
                            feature_prob = cond_prob
                            break
                    # the probability of the sample belonging to the current class
                    # is the probability of all of the feature values 
                    class_prob *= feature_prob
                class_probs[class_] = class_prob
            # normalize class probabilities
            # to avoid NaN values, all the probabilities will remain the same
            if np.all(class_probs, 0):
                #print(f"found all probabilities 0 at image {i}")
                class_probs += 1
            total = np.sum(class_probs)
            posterior_probs = class_probs / total
            probs[i] = posterior_probs.reshape((5,))
        return probs

    def predict_class(self, X):
        probs = self.predict_probabilities(X)
        # return class with highest probability for each sample
        return np.argmax(probs, axis=1)