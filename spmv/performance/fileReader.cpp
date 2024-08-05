#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::string filePath = "/home/jvalglz/gptFortranLara/spmv/matrix_csr_test_final.txt";
    std::vector<std::string> searchStrings = {"# Values:", "# Column Indices:", "# Row Pointers:"};

    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 1;
    }

    std::string line;
        while (getline(file, line) && !searchStrings.empty()) {
        // Check if the current line contains the first search string
        if (line.find(searchStrings.front()) != std::string::npos) {
            std::cout << "worked: " << searchStrings.front() << std::endl; // Print the found search string
            searchStrings.erase(searchStrings.begin()); // Remove the found string from the vector
        }
    }

    file.close();
    return 0;
}