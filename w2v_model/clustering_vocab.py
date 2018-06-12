from gensim.models import Word2Vec
import numpy as np  # Make sure that numpy is imported


def make_feature_vec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    feature_vec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            feature_vec = np.add(feature_vec, model[word])
    #
    # Divide the result by the number of words to get the average
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_feature_vecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        #
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))
        #
        # Call the function (defined above) that makes average feature vectors
        review_feature_vecs[counter] = make_feature_vec(review, model, \
                                                    num_features)
        #
        # Increment the counter
        counter = counter + 1.
    return review_feature_vecs


def k_means_clustering(model):
    from sklearn.cluster import KMeans
    import time

    start = time.time()  # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.wv.syn0
    num_clusters = 300  # word_vectors.shape[0] / 7
    print "Working on {} clusters...".format(num_clusters)

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."

    # Create a Word / Index dictionary, mapping each vocabulary word to
    word_centroid_map = dict(zip(model.wv.index2word, idx))

    # For the first clusters
    for cluster in range(num_clusters):
        print "\nCluster %d" % cluster
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in range(len(word_centroid_map.values())):
            if word_centroid_map.values()[i] == cluster:
                words.append(word_centroid_map.keys()[i])
        print words


if __name__ == '__main__':
    model = Word2Vec.load("300features_10minwords_10context")

    k_means_clustering(model)
