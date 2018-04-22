#include "Lanczos.h"
#include "Timer.h"

extern "C" {
    // Solve A*x = b where A is tridiagonal.
    void dgtsv_(
        int* N, int* NRHS, double* DL, double* D, double* DU, double* B,
        int* LDB, int* info);
}

void reference_lanczos(MajoranaKitaevHamiltonian& H,
                       double mu,
                       int lanczos_steps,
                       RealVec& alpha,
                       RealVec& beta,
                       boost::random::mt19937* gen) {
    Vec initial_state = get_random_state(Space::from_majorana(H.mN), gen);
    reference_lanczos(H, mu, lanczos_steps, alpha, beta, initial_state);
}

// Method from Cullum and Willoughby, Lanczos algorithms for large
// symmetric eigenvalue computations (section 2.1).
void reference_lanczos(MajoranaKitaevHamiltonian& H,
                       double mu,
                       int lanczos_steps,
                       RealVec& alpha,
                       RealVec& beta_result,
                       Vec& initial_state) {
    assert(abs(initial_state.norm() - 1.) < epsilon);

    int D = H.dim();
    SpMat Id(D, D);
    Id.setIdentity();
    SpMat A = H.matrix + mu * Id;

    int m = lanczos_steps;

    alpha = RealVec(m);
    beta_result = RealVec(m-1);

    RealVec beta(m+1);
    beta(0) = 0.;

    // v_{i-1} = v_0
    Vec vi_minus_1 = Vec::Zero(D); 

    // v_1 = v_i
    Vec vi = initial_state;   

    for (int i = 1; i <= m; i++) {
        Vec u = A * vi;
        u += - beta(i-1) * vi_minus_1;
        alpha(i-1) = u.dot(vi).real();
        u += - alpha(i-1) * vi;
        beta(i) = u.norm();

        vi_minus_1 = vi;
        vi = u / beta(i);
    }

    for (int i = 0; i < m-1; i++) {
        beta_result(i) = beta(i + 1);
    }
}

void factorized_lanczos(FactorizedHamiltonianGenericParity& H,
                        double mu,
                        int lanczos_steps,
                        boost::random::mt19937* gen,
                        RealVec& alpha,
                        RealVec& beta,
                        bool extended_beta
                        ) {
    Mat initial_state = get_factorized_random_state(H.space, gen);
    factorized_lanczos(H, mu, lanczos_steps, initial_state,
                       alpha, beta, extended_beta);
}

void factorized_lanczos(FactorizedHamiltonianGenericParity& H,
                        double mu,
                        int lanczos_steps,
                        Mat& initial_state,
                        RealVec& alpha,
                        RealVec& beta_result,
                        bool extended_beta
                        ) {
    assert(abs(initial_state.norm() - 1.) < epsilon);

    int m = lanczos_steps;
    alpha = RealVec(m);
    beta_result = RealVec(extended_beta ? m : m-1);

    RealVec beta(m+1);
    beta(0) = 0.;

    // v_{i-1} = v_0
    Mat vi_minus_1 = Mat::Zero(H.space.left.D, H.space.right.D); 

    // v_i = v_1
    Mat vi = initial_state;

    for (int iter = 1; iter <= m; iter++) {
        // u = A * vi - beta(iter-1) * vi_minus_1 + mu * vi
        Mat u = Mat::Zero(vi.rows(), vi.cols());
        H.act(u, vi);

        // this gives the wrong answer!
        // u += - beta(iter-1) * vi_minus_1 + mu * vi;
        u += - beta(iter-1) * vi_minus_1;
        u += mu * vi;

        // alpha(iter-1) = u.dot(vi).real();
        alpha(iter-1) = u.conjugate().cwiseProduct(vi).sum().real();

        u -= alpha(iter-1) * vi;

        beta(iter) = u.norm();

        vi_minus_1 = vi;
        vi = u / beta(iter);
    }

    for (int i = 0; i < beta_result.size(); i++) {
        beta_result(i) = beta(i + 1);
    }
}

static int count_appearances(double x, RealVec& v) {
    int n = 0;

    for (int i = 0; i < v.size(); i++) {
        if (fabs(x - v(i)) < epsilon) {
            n++;
        }
    }

    return n;
}

static bool element_of(double x, RealVec& v) {
    return count_appearances(x, v) > 0;
}

static bool multiple_element_of(double x, RealVec& v) {
    return count_appearances(x, v) > 1;
}

RealVec all_lanczos_evs(const RealVec& alpha, const RealVec& beta) {
    assert(beta.size() == alpha.size() - 1);
    SelfAdjointEigenSolver<RealMat> solver;
    solver.computeFromTridiagonal(alpha, beta, EigenvaluesOnly);
    return solver.eigenvalues();
}

// T_hat = T with first row and first column removed
static void get_T_hat(const RealVec& alpha, const RealVec& beta,
                      RealVec& T_hat_alpha, RealVec& T_hat_beta) {
    assert(alpha.size() > 1);
    int m = alpha.size();
    T_hat_alpha = RealVec(m - 1);
    T_hat_beta = RealVec(m - 2);

    for (int i = 0; i < m-1; i++) {
        T_hat_alpha(i) = alpha(i+1);

        if (i < m-2) {
            T_hat_beta(i) = beta(i+1);
        }
    }
}

RealVec find_good_lanczos_evs(const RealVec& alpha,
                              const RealVec& beta) {
    assert(beta.size() == alpha.size() - 1);

    if (alpha.size() == 1) {
        // Don't bother with a single step case
        return RealVec(0);
    }
    
    int m = alpha.size();

    RealVec T_hat_alpha;
    RealVec T_hat_beta;
    get_T_hat(alpha, beta, T_hat_alpha, T_hat_beta);

    RealVec T_evs = all_lanczos_evs(alpha, beta);
    RealVec T_hat_evs = all_lanczos_evs(T_hat_alpha, T_hat_beta);

    assert(T_evs.size() == m);
    assert(T_hat_evs.size() == m-1);

    RealVec good_evs(m);
    int num_good_evs = 0;

    for (int i = 0; i < m; i++) {
        double lambda = T_evs(i);

        //
        // We accept an eigenvalue as good if:
        // 1. It is a multiple eigenvalue of T, or if
        // 2. It is a single eigenavlue of T,
        //    and is not an eigenvalue of T_hat
        //
        if (multiple_element_of(lambda, T_evs) ||
            !element_of(lambda, T_hat_evs)) {

            good_evs(num_good_evs++) = lambda;
        }
    }

    RealVec just_good_evs = get_head(good_evs, num_good_evs);
    RealVec unique_good_evs = get_unique_elements(just_good_evs, epsilon);
    return unique_good_evs;
}

// Return both eigenvalues and eigenvectors of the Lanczos matrix:
// T = V.D.V^{-1}, and eigenvectors = V.
RealVec all_lanczos_evs(const RealVec& alpha,
                        const RealVec& beta,
                        RealMat& eigenvectors) {
    assert(beta.size() == alpha.size() - 1);
    SelfAdjointEigenSolver<RealMat> solver;
    solver.computeFromTridiagonal(alpha, beta, ComputeEigenvectors);
    eigenvectors = solver.eigenvectors();
    return solver.eigenvalues();
}

// Whether v has an element that is within eps of x.
static bool has_close_element(double x, vector<double> v, double eps) {
    for (int i = 0; i < v.size(); i++) {
        if (abs(x - v.at(i)) < eps) {
            return true;
        }
    }

    return false;
}

// extended_beta needs to have length m, and its last element should be
// the extra beta value that is not used in the Lanczos matrix..
RealVec find_good_lanczos_evs_and_errs_full_diagonalization(
        const RealVec& alpha,
        const RealVec& extended_beta,
        RealVec& error_estimates) {

    assert(extended_beta.size() == alpha.size());

    if (alpha.size() == 1) {
        // Don't bother with a single step case
        error_estimates = RealVec(0);
        return RealVec(0);
    }

    int m = alpha.size();

    RealVec beta = get_head(extended_beta, m-1);
    double last_beta = extended_beta(m-1);

    RealVec T_hat_alpha;
    RealVec T_hat_beta;
    get_T_hat(alpha, beta, T_hat_alpha, T_hat_beta);

    RealMat T_evectors;
    RealVec T_evs = all_lanczos_evs(alpha, beta, T_evectors);
    RealVec T_hat_evs = all_lanczos_evs(T_hat_alpha, T_hat_beta);

    assert(T_evs.size() == m);
    assert(T_hat_evs.size() == m-1);

    vector<double> spurious_evs;
    RealVec good_evs(m);
    int num_good_evs = 0;

    // Find the good eigenvalues
    for (int i = 0; i < m; i++) {
        double lambda = T_evs(i);

        if (multiple_element_of(lambda, T_evs)) {
            // Multiple evs of T are good
            good_evs(num_good_evs++) = lambda;
        }
        else if (!element_of(lambda, T_hat_evs)) {
            // evs of T but not of T_hat are also good
            good_evs(num_good_evs++) = lambda;
        }
        else {
            spurious_evs.push_back(lambda);
        }
    }

    // Compute absolute error estimates
    error_estimates = RealVec(num_good_evs);

    for (int i = 0; i < num_good_evs; i++) {
        double lambda = good_evs(i);

        if (multiple_element_of(lambda, T_evs)) {
            // Multiple evs of T have no error
            // (they converged because they will be evs of
            // all bigger Lanczos matrices).
            error_estimates(i) = 0.;
        }
        else if (has_close_element(good_evs(i), spurious_evs, epsilon)) {
            // If there is a nearby spurious ev, then the good
            // ev has negligible error.
            error_estimates(i) = 0.;
        }
        else {
            // Otherwise, the eigenvectors give an estimate
            // of the error. Formula from Lemma 2.3.5 of
            // Cullum & Willoughby (eq. 2.3.29).
            error_estimates(i) = 2.5 * abs(last_beta * T_evectors(m-1, i));
        }
    }

    RealVec just_good_evs = get_head(good_evs, num_good_evs);
    RealVec unique_good_evs = get_unique_elements(just_good_evs,
                                                  epsilon);

    return unique_good_evs;
}

RealMat get_tridiagonal_matrix(const RealVec& diagonal,
                               const RealVec& subdiagonal) {
    int N = diagonal.size();
    RealMat mat = RealMat::Zero(N, N);

    for (int i = 0; i < N; i++) {
        mat(i, i) = diagonal(i);

        if (i < N-1) {
            mat(i, i+1) = subdiagonal(i);
            mat(i+1, i) = subdiagonal(i);
        }
    }

    return mat;
}

RealSpMat get_tridiagonal_sparse_matrix(const RealVec& diagonal,
                                        const RealVec& subdiagonal) {
    int N = diagonal.size();
    vector<RealTriplet> triplets;

    for (int i = 0; i < N; i++) {
        triplets.push_back(RealTriplet(i, i, diagonal(i)));

        if (i < N-1) {
            triplets.push_back(RealTriplet(i, i+1, subdiagonal(i)));
            triplets.push_back(RealTriplet(i+1, i, subdiagonal(i)));
        }
    }

    RealSpMat mat = RealSpMat(N, N);
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
}

bool is_eigenvector(const RealVec& alpha,
                    const RealVec& beta,
                    double ev,
                    const RealVec& evec) {
    RealSpMat mat = get_tridiagonal_sparse_matrix(alpha, beta);
    RealVec result = mat * evec;

    for (int i = 0; i < alpha.size(); i++) {
        if (abs(result(i) - ev * evec(i)) > epsilon) {
            return false;
        }
    }

    return true;
}

// Use inverse iteration. Acting on an initial vector v
// with (T - ev)^{-1}, the piece of v that corresponds to the eigenvector
// closest to ev gets amplified. We repeat until we find that v
// is an eigenvector.
RealVec find_eigenvector_for_ev(const RealVec& alpha,
                                const RealVec& beta,
                                double ev,
                                boost::random::mt19937* gen,
                                int max_iterations) {
    int N = alpha.size();

    // beta can have extra values at the end, and these will be ignored
    assert(beta.size() >= N-1);

    int NRHS = 1;

    // Lower subdiagonal
    RealVec DL(N-1);

    // Diagonal
    RealVec D(N);

    // Upper subdiagonal
    RealVec DU(N-1);

    // The initial vector in A*x=v. Will contain x after each solution.
    RealVec v = get_random_real_vector(N, gen);

    int iteration = 0;

    while (iteration < max_iterations) {
        DL = beta;
        DU = beta;

        for (int i = 0; i < N; i++) {
            D(i) = alpha(i) - ev + epsilon;
        }
    
        int info;
    
        dgtsv_(&N, &NRHS, DL.data(), D.data(), DU.data(), v.data(),
               &N, &info);

        if (info) {
            cerr << "dgtsv error: " << info << endl;
            exit(1);
        }

        v = v / v.norm();
        iteration++;

        if (is_eigenvector(alpha, beta, ev, v)) {
            break;
        }
    }

    return v;
}

// extended_beta needs to have length m, and its last element should be
// the extra beta value that is not used in the Lanczos matrix..
RealVec find_good_lanczos_evs_and_errs(const RealVec& alpha,
                                       const RealVec& extended_beta,
                                       RealVec& error_estimates,
                                       boost::random::mt19937* gen) {
    assert(extended_beta.size() == alpha.size());

    if (alpha.size() == 1) {
        // Don't bother with a single step case
        error_estimates = RealVec(0);
        return RealVec(0);
    }
    
    int m = alpha.size();

    RealVec beta = get_head(extended_beta, m-1);
    double last_beta = extended_beta(m-1);

    RealVec T_hat_alpha;
    RealVec T_hat_beta;
    get_T_hat(alpha, beta, T_hat_alpha, T_hat_beta);

    RealVec T_evs = all_lanczos_evs(alpha, beta);
    RealVec T_hat_evs = all_lanczos_evs(T_hat_alpha, T_hat_beta);

    assert(T_evs.size() == m);
    assert(T_hat_evs.size() == m-1);

    vector<double> spurious_evs;
    RealVec good_evs(m);
    int num_good_evs = 0;

    // Find the good eigenvalues
    for (int i = 0; i < m; i++) {
        double lambda = T_evs(i);

        if (multiple_element_of(lambda, T_evs)) {
            // Multiple evs of T are good
            good_evs(num_good_evs++) = lambda;
        }
        else if (!element_of(lambda, T_hat_evs)) {
            // evs of T but not of T_hat are also good
            good_evs(num_good_evs++) = lambda;
        }
        else {
            spurious_evs.push_back(lambda);
        }
    }

    RealVec just_good_evs = get_head(good_evs, num_good_evs);
    RealVec unique_good_evs =
        get_unique_elements(just_good_evs, epsilon);

    // Compute absolute error estimates
    error_estimates = RealVec(unique_good_evs.size());

    for (int i = 0; i < unique_good_evs.size(); i++) {
        double lambda = unique_good_evs(i);

        if (multiple_element_of(lambda, T_evs)) {
            // Multiple evs of T have no error
            // (they converged because they will be evs of
            // all bigger Lanczos matrices).
            error_estimates(i) = 0.;
        }
        else if (has_close_element(unique_good_evs(i),
                                   spurious_evs, epsilon)) {
            // If there is a nearby spurious ev, then the good
            // ev has negligible error.
            error_estimates(i) = 0.;
        }
        else {
            // Otherwise, the eigenvectors give an estimate
            // of the error. Formula from Lemma 2.3.5 of
            // Cullum & Willoughby (eq. 2.3.29).
            RealVec T_evec = find_eigenvector_for_ev(
                alpha, extended_beta, lambda, gen);
            assert(T_evec.size() == m);
            error_estimates(i) = 2.5 * abs(last_beta * T_evec(m-1));
        }
    }

    return unique_good_evs;
}

void print_lanczos_results(RealVec& H_evs, RealVec& lanczos_evs) {
    RealVec unique_A_evs = get_unique_elements(H_evs, epsilon);

    cout << "i\tH ev\t\tgood ev\t\trelative error\n";

    int j = 0;

    for (int i = 0; i < unique_A_evs.size(); i++) {
        cout << i << "\t" << unique_A_evs(i);

        if (i < lanczos_evs.size()) {
            cout << "\t" << lanczos_evs(i)
                    << "\t (" << relative_err(lanczos_evs(i),
                                              unique_A_evs(i))
                    << " )";
        }

        if (j < lanczos_evs.size() && 
            fabs(lanczos_evs(j) - unique_A_evs(i)) < epsilon) {

            j++;
        }

        cout << "\n";
    }
    cout << endl;

    if (j == lanczos_evs.size()) {
        cout << "All good T values match A values.\n\n";
    }
    else {
        cout << "Some T values don't match A values!!\n\n";
    }

    double match_ratio = (double) j / (double) unique_A_evs.size();
    cout << "Matched " << match_ratio * 100.
         << " % of total eiganvalues\n\n";
}

map<int, int> find_nearest_true_ev(RealVec& true_evs, RealVec& lanczos_evs) {
    map<int, int> nearest;

    for (int j = 0; j < lanczos_evs.size(); j++) {
        int best_i = 0;
        double best_dist = abs(lanczos_evs(j) - true_evs(best_i));

        for (int i = 0; i < true_evs.size(); i++) {
            double dist = abs(lanczos_evs(j) - true_evs(i));

            if (dist < best_dist) {
                best_dist = dist;
                best_i = i;
            }
        }

        nearest[best_i] = j;
    }

    return nearest;
}

void print_lanczos_results(RealVec& H_evs,
                           RealVec& lanczos_evs,
                           RealVec& error_estimates) {
    RealVec unique_A_evs = get_unique_elements(H_evs, epsilon);
    map<int, int> nearest = find_nearest_true_ev(unique_A_evs, lanczos_evs);

    cout << "i\tH ev\t\tgood ev\t\trel-err\t\tabs-err\t\terr-est\n";

    for (int i = 0; i < unique_A_evs.size(); i++) {
        cout << i << "\t" << unique_A_evs(i);

        if (nearest.count(i)) {
            int j = nearest.at(i);
            double rel_err = relative_err(lanczos_evs(j), unique_A_evs(i));
            double abs_err = abs(lanczos_evs(j) - unique_A_evs(i));

            cout << "\t" << lanczos_evs(j)
                 << "\t" << rel_err
                 << "\t" << abs_err
                 << "\t" << error_estimates(j);
        }

        cout << "\n";
    }

    cout << endl;
}
