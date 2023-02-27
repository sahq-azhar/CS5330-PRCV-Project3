#include "csvHelp.h"



bool write_to_file(std::string file_name, std::string label, std::vector<double> features) {

	std::ofstream file;
	file.open(file_name, std::ios::app);
	file << label << ",";
	for (double feature : features)
		file << feature << ",";
	file << std::endl;

	file.close();

	return true;
}



bool write_confusion_matrix_to_file(string file_name, set<string>& s, map<string, map<string, int>>& cm) {

	vector<string> labels(s.size());
	std::copy(s.begin(), s.end(), labels.begin());
	std::ofstream file;
	file.open(file_name, std::ios::app);
	file << "*,";
	for (string label : labels)
		file << label << ",";
	file << endl;
	for (string pred_label : labels) {
		file << pred_label << ",";
		for (string real_label : labels) {
			file << cm[pred_label][real_label] << ",";
		}
		file << endl;
	}

	file.close();

	return true;

}



bool read_from_file(std::string file_name, std::vector<std::string>& labels, std::vector<std::vector<double>>& nfeatures) {

	int size = 7;
	std::ifstream file;
	file.open(file_name);
	std::string label;
	std::string element;
	while (getline(file, label, ',')) {

		std::vector<double> features;

		for (int i = 0; i < size - 1; i++) {
			getline(file, element, ',');
			features.push_back(std::stod(element));
		}
		getline(file, element, '\n');
		features.push_back(std::stod(element));
		nfeatures.push_back(features);
		labels.push_back(label);
	}

	file.close();

	return true;
}


