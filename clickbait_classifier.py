from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load our data into two Python lists
with open("clickbait.txt") as f:
    lines = f.read().strip().split("\n")
    lines = [line.split("\t") for line in lines]
headlines, labels = zip(*lines)


print("Let's look at the first few examples")
print(headlines[:5])
print(labels[:5])

# How big is our dataset?
print("How big is our dataset?")
print(len(headlines))


# Break dataset into test and train python
train_headlines = headlines[:8000]
test_headlines = headlines[8000:]

train_labels = labels[:8000]
test_labels = labels[8000:]

# Create a vectorizer and classifier
vectorizer = TfidfVectorizer()
svm = LinearSVC()

# Transform our text data into numerical vectors
train_vectors = vectorizer.fit_transform(train_headlines)
test_vectors = vectorizer.transform(test_headlines)

# Train the classifier and predict on test set
svm.fit(train_vectors, train_labels)
predictions = svm.predict(test_vectors)

print("First few test headlines")
print(test_headlines[0:5])
print("Our classifiers predictions for these")
print(predictions[:5])
print("The actual labels to compare to our classfiers predictions")
print(test_labels[:5])

print("The overall accuracy score")
print(accuracy_score(test_labels, predictions))

new_headlines = ["10 Cities That Every Hipster Will Be Moving To Soon", 'Vice President Mike Pence Leaves NFL Game Saying Players Showed "Disrespect" Of Anthem, Flag']

print("the new headlines")
print(new_headlines)
new_vectors = vectorizer.transform(new_headlines)

print("the new predictions")
new_predictions = svm.predict(new_vectors)
print(new_predictions)
