#include "eigen_utils.h"
#include <boost/random/normal_distribution.hpp>

Vec get_random_vector(int size, boost::random::mt19937* gen) {
    Vec vec(size);
    boost::random::normal_distribution<> dist(0., 1.);

    for (int i = 0; i < vec.size(); i++) {
        vec(i) = cpx(dist(*gen), dist(*gen));
    }

    return vec;
}

RealVec get_random_real_vector(int size, boost::random::mt19937* gen) {
    RealVec vec(size);
    boost::random::normal_distribution<> dist(0., 1.);

    for (int i = 0; i < vec.size(); i++) {
        vec(i) = dist(*gen);
    }

    return vec;
}

Mat get_random_matrix(int rows, int cols, boost::random::mt19937* gen) {
    Mat matrix(rows, cols);
    boost::random::normal_distribution<> dist(0., 1.);

    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            matrix(i, j) = cpx(dist(*gen), dist(*gen));
        }
    }

    return matrix;
}

void write_matrix_binary(ofstream& out, const Mat& matrix) {
    Mat::Index rows = matrix.rows();
    Mat::Index cols = matrix.cols();

    out.write((char*) (&rows), sizeof(Mat::Index));
    out.write((char*) (&cols), sizeof(Mat::Index));
    out.write((char*) matrix.data(), rows * cols * sizeof(Mat::Scalar));
}

Mat read_matrix_binary(ifstream& in) {
    Mat::Index rows;
    Mat::Index cols;

    in.read((char*) (&rows), sizeof(Mat::Index));
    in.read((char*) (&cols), sizeof(Mat::Index));

    Mat matrix(rows, cols);
    in.read((char*) matrix.data(),
            rows * cols * sizeof(Mat::Scalar));

    return matrix;
}

void write_vector_binary(ofstream& out, const Vec& vec) {
    Vec::Index size = vec.size();
    out.write((char*) (&size), sizeof(Vec::Index));
    out.write((char*) vec.data(), size * sizeof(Vec::Scalar));
}

Vec read_vector_binary(ifstream& in){
    Vec::Index size;
    in.read((char*) (&size), sizeof(Vec::Index));

    Vec vec(size);
    in.read((char*) vec.data(), size * sizeof(Vec::Scalar));

    return vec;
}

void write_real_vector_binary(ofstream& out, const RealVec& vec) {
    RealVec::Index size = vec.size();
    out.write((char*) (&size), sizeof(RealVec::Index));
    out.write((char*) vec.data(), size * sizeof(RealVec::Scalar));
}

RealVec read_real_vector_binary(ifstream& in){
    RealVec::Index size;
    in.read((char*) (&size), sizeof(RealVec::Index));

    RealVec vec(size);
    in.read((char*) vec.data(), size * sizeof(RealVec::Scalar));

    return vec;
}

RealVec get_head(const RealVec& v, int n) {
    RealVec head(n);

    for (int i = 0; i < n; i++) {
        head(i) = v(i);
    }

    return head;
}

RealVec get_unique_elements(RealVec& v, double unique_eps) {
    RealVec uniques(v.size());
    int num_uniques = 0;

    if (v.size() == 0) {
        return uniques.head(0);
    }

    uniques(num_uniques++) = v(0);

    for (int i = 1; i < v.size(); i++) {
        if (fabs(uniques(num_uniques-1) - v(i)) > unique_eps) {
            uniques(num_uniques++) = v(i);
        }
    }

    // Remove the padding
    RealVec just_uniques = get_head(uniques, num_uniques);
    return just_uniques;
}

void add_nonzeros_to_triplets(vector<CpxTriplet>& triplets, const Mat& mat,
                              int row_offset, int col_offset) {
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            if (mat(i, j) != cpx(0., 0.)) {
                triplets.push_back(
                    CpxTriplet(i + row_offset,
                               j + col_offset,
                               mat(i, j)));
            }
        }
    }
}
