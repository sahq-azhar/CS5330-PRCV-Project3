// The following code implements the K-Nearest Neighbor (KNN) algorithm for classification.

#include "k-near.h"
#include "csvHelp.h"

// Data class to store each data point's label, features, and distance to the query point.
class Data {
public:
	double distance;
	string label;
	vector<double> features;
	Data() : distance(0) {}
};

// Function to calculate the Euclidean distance between two feature vectors.
double euclideanDistance(const vector<double>& f1, const vector<double>& f2) {
	double sum = 0;
	for (size_t i = 0; i < f1.size(); i++)
		sum += pow((f1[i] - f2[i]), 2);
	return sqrt(sum);
}

// Comparator function to sort the Data objects based on their distances in ascending order.
bool cmp(const Data& a, const Data& b) {
	return a.distance < b.distance;
}

// Function to calculate the Manhattan distance between two feature vectors.
double manhattanDistance(const vector<double>& f1, const vector<double>& f2) {
	double sum = 0;
	for (size_t i = 0; i < f1.size(); i++)
		sum += abs(f1[i] - f2[i]);
	return sum;
}

// Function to fill the distances between all points and the query point, and store them in Data objects.
void fillDistances(const vector<vector<double>>& query, const vector<vector<double>>& nfeatures, const vector<string>& labels, vector<Data>& data) {
	for (size_t i = 0; i < labels.size(); i++) {
		Data data_point;
		data_point.label = labels[i];
		data_point.features = nfeatures[i];
		data_point.distance = euclideanDistance(data_point.features, query[0]);
		data.push_back(data_point);
	}
}

// Function to perform KNN classification given the query point and value of K.
string knn_classifier(vector<vector<double>>& query, int k) {
	// Read the training feature database from a CSV file.
	string fileName = "imgtrainingFeatures_data.csv";
	std::vector<std::string> labels;
	std::vector<std::vector<double>> nfeatures;
	read_from_file(fileName, labels, nfeatures);

	// Create a vector of Data objects and reserve space for efficiency.
	vector<Data> data;
	data.reserve(labels.size());

	// Fill the distances between all points and the query point.
	fillDistances(query, nfeatures, labels, data);

	// Sort the Data objects based on their distances in ascending order.
	sort(data.begin(), data.end(), cmp);

	// Check if the closest point is too far away, and return "unknown" if so.
	if (data[0].distance > 10)
		return "unknown";

	// Use a map to count the occurrence of each label in the K closest points.
	map<string, int> count;
	int max = -1;
	string mode_label;

	for (size_t i = 0; i < k; i++) {
		count[data[i].label] += 1;
		if (count[data[i].label] > max) {
			max = count[data[i].label];
			mode_label = data[i].label;
		}
	}

	// Return the label that occurred most frequently among the K closest points.
	return mode_label;
}