import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

olivetti_dataset = fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0,
download_if_missing=True)

lfw_people_dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4,download_if_missing=True)

#change to the name of the dataset you want to use
x = olivetti_dataset.data
y = olivetti_dataset.target

n_samples, h, w = olivetti_dataset.images.shape

#training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y)

n_components=100
print("Extracting eigenfaces: ")
pca = PCA(n_components=n_components).fit(x_train)
eigenfaces = pca.components_


def plot_gallery(images, titles, h, w, n_row=3, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# plot the result of the prediction on a portion of the test set
# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()

# print(lfw_people_dataset)
