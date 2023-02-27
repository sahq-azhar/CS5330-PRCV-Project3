#pragma once
class csvHelp
{
};

#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <set>
using namespace std;


bool write_to_file(std::string file_name, std::string label, std::vector<double> features);
bool write_confusion_matrix_to_file(string file_name, set<string>& labels, map<string, map<string, int>>& cm);
bool read_from_file(std::string file_name, std::vector<std::string>& labels, std::vector<std::vector<double>>& nfeatures);
