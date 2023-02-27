#pragma once
class knn_classifier
{
};

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <map>

using namespace std;

string knn_classifier(vector<vector<double>>& query, int k);