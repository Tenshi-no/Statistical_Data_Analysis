#include <cmath>
#include <vector>
#include <random>
#include <iomanip> //��������� ��� ������ setprecision
#include <iostream>
#include <fstream>
using namespace std;

double NormalRand(double mean, double std_dev)
{
    static mt19937 gen(1);

    normal_distribution<double> dist(mean, std_dev);
    return dist(gen);
}

// �������
double TrueFoo(double x)
{
    double part1 = sin(x);
    double part2 = x / 10.0;
    double part3 = 50.0 / log(x + 10.0);
    double result = part1 + part2 + part3;
    return result;
}

double Equation(double x)
{
    double noise = NormalRand(1.0, 0.05);
    return TrueFoo(x)*noise;
}

// �������������� ����
double parabolic_kernel(double z)
{
    if (abs(z) <= 1.0)
    {
        return 0.75 * (1.0 - z * z);
    }
    else
    {
        return 0.0;
    }
}

// � - ������ ������� ��������, � - ������ �������� ��������
// ������ ��������� ������� ����������� ��������
double predict_leaveoo(const vector<double>& X, const vector<double>& Y, int j, double h)
{
    double numerator = 0.0;
    double denominator = 0.0;
    int n = X.size();

    for (int i = 0; i < n; ++i)
    {
        if (i == j)
        {
            continue; // ��������� j-�� �����
        }
        double z = (X[j] - X[i]) / h;
        double weight = parabolic_kernel(z);
        numerator += Y[i] * weight;
        denominator += weight;
    }

    if (denominator < 0.00000000001)
    {
        double sum = 0.0;
        int cnt = 0;
        for (int i = 0; i < n; ++i)
        {
            if (i != j)
            {
                sum += Y[i];
                cnt++;
            }
        }
        if (cnt > 0)
        {
            return sum / cnt;
        }
        else return 0;
    }

    return numerator / denominator;
}

// ������� ���������� ������
double calculate_E(const vector<double>& X, const vector<double>& Y, double h)
{
    int n = X.size(); // ����� �������� ������ �������
    double result_error = 0.0;

    for (int j = 0; j < n; ++j)
    {
        double y_pred = predict_leaveoo(X, Y, j, h);
        double error = Y[j] - y_pred;
        result_error += error * error;
    }
    return result_error / n;
}

// ������������ �� ���� ������ ��� ���������� ������ ���������
double predict_all(const vector<double>& X, const vector<double>& Y, double x_query, double h)
{
    double numerator = 0.0;
    double denominator = 0.0;

    for (int i = 0; i < X.size(); ++i)
    {
        double z = (x_query - X[i])/h;
        double weight = parabolic_kernel(z);
        numerator += Y[i] * weight;
        denominator += weight;
    }
    if (denominator == 0.0)
    {
        denominator = 0.00001;
    }
    return numerator/denominator;
}


//���� �������� ������������

int main()
{
    int N = 1000;
    vector<double> X(N);
    vector<double> Y(N);
    vector<double> Y_true(N);

    // 1. ��������� ������
    for (int i = 1; i <= N; ++i)
    {
        double xi = i * 0.1;
        X[i - 1] = xi;
        Y[i - 1] = Equation(xi);
        Y_true[i-1] = TrueFoo(xi);
//        cout << i << " " << Y_true[i-1] << endl;
    }


    // 2. ����� ������������ h
    double n = pow(N, -0.2);
    cout << "Value C\t\t" << "Value h\t\t" << "MSE\t\t" << "\n";
    double best_c = 0.2;
    double min_error = pow(10, 2);

    for (double c = 0.2; c <= 10; c += 0.1)
    {
        double h = c * n;
        double error = calculate_E(X, Y, h);
        cout << fixed << setprecision(2) << c << "\t\t"
             << setprecision(4) << h << "\t\t"
             << setprecision(5) << error << "\n";

        if (error < min_error)
        {
            min_error = error;
            best_c = c;
        }
    }
    double best_h = best_c * n;

    cout << "Results" << endl;
    cout << "Best c: " << best_c << endl;
    cout << "Best h: " << best_h << endl;
    cout << "MSE: " << min_error << endl;
    cout << endl;

    ofstream f1("part1_data.csv");
    f1 << "x, y_true, y_noisy, y_pred\n";
    for (int i = 0; i < N; ++i)
    {
        double y_pred = predict_all(X, Y, X[i], best_h);
        f1 << X[i] << ", " << Y_true[i] << ", " << Y[i] << ", " << y_pred << "\n";
    }
    f1.close();

    ofstream f2("part1_error.csv");
    f2 << "c,h,error\n";
    for (double c = 0.2; c <= 10; c += 0.1)
    {
        double h = c * n;
        double error = calculate_E(X, Y, h);
        f2 << c << "," << h << "," << error << "\n";
    }
    f2.close();

    // ������� 5 ��������� ����� ��� ��������
    vector<int> test_indices = {10, 250, 500, 750, 990};
    cout << "Value X" << "\t\t" << "Noise Y" << "\t\t" << "Pred Y" << "\t\t" << "True Y" << "\t\t" << endl;
    for (int i = 0; i < test_indices.size(); ++i)
    {
        int idx = test_indices[i];
        double x_val = X[idx];
        double y_noisy = Y[idx];

        double y_pred = predict_all(X, Y, x_val, best_h);

        double y_true = Y_true[idx];
//        double relative_error = abs(y_pred - y_true) / abs(y_true) * 100.0;
        cout << fixed << setprecision(2) << x_val << "\t\t"
             << setprecision(4) << y_noisy << "\t\t"
             << setprecision(4) << y_pred << "\t\t"
             << setprecision(4) << y_true << "\t\t" << endl;

//        cout << x_val << "\t\t" << y_noisy << "\t\t" << y_pred << "\t\t" << y_true << "\t\t" << endl;
    }

    cout << "=============PART2==========" << endl;

    vector<double> X_real =
    {
        35.0, 36.7, 36.7, 36.8, 36.8, 37.1, 37.5, 38.0, 38.2, 38.7,
        38.9, 39.0, 39.1, 39.5, 39.8, 40.2, 40.5, 41.0, 41.3, 41.3,
        41.5, 41.8, 42.0, 42.4, 42.8, 42.9, 43.1, 43.3, 43.5, 43.8,
        44.0, 44.2, 44.5, 44.8, 45.0, 45.2, 45.4, 45.6, 45.7, 46.0,
        46.2, 46.5, 46.8, 47.0, 47.3, 47.5, 47.8, 48.0, 48.5, 49.0
    };

    vector<double> Y_real =
    {
        6992000, 7162350, 7150000, 7176000, 7189000, 7240350, 7300000, 7350000, 7380000, 7450000,
        7480000, 7500000, 7550000, 7601100, 7638650, 7679060, 7679940, 7703290, 7721230, 7745080,
        7750000, 7762520, 7800000, 7850000, 7890000, 7900000, 7950000, 7980000, 8000000, 8050000,
        8097160, 8120000, 8180000, 8200000, 8250000, 8300000, 8350000, 8443600, 8464500, 8500000,
        8550000, 8600000, 8650000, 8700000, 8750000, 8800000, 8850000, 8892000, 8911500, 9067500
    };
    int N_real = X_real.size();
    double n_real = pow(N_real, -0.2);
    cout << "n_real = " << n_real << endl;
    cout << endl;
    double best_c_real = 0.2;
    double min_error_real = pow(10, 100);
    cout << "REAL C\t\t" << "REAL h\t\t" << "REAL ERROR" << endl;
    for (double c = 0.2; c <= 10.0; c += 0.1)
    {
        double h_real = c * n_real;
        double error_real = calculate_E(X_real, Y_real, h_real);

        cout << fixed << setprecision(2) << c << "\t\t"
             << setprecision(4) << h_real << "\t\t"
             << setprecision(2) << error_real << "\n";

        if (error_real < min_error_real)
        {
            min_error_real = error_real;
            best_c_real = c;
        }
    }

    double best_h_real = best_c_real * n_real;

    cout << "RESULT_PART2" << endl;
    cout << "Best c = " << best_c_real << endl;
    cout << "h = " << best_h_real <<endl;
    cout << "MSE = " << min_error_real << endl;
//        double relative_error = abs(y_pred - y_true) / abs(y_true) * 100.0;
    ofstream f3("part2_data.csv");
    f3 << "x,y_noisy,y_pred\n";
    for (int i = 0; i < N_real; ++i)
    {
        double y_pred = predict_all(X_real, Y_real, X_real[i], best_h_real);
        f3 << X_real[i] << "," << Y_real[i] << "," << y_pred << "\n";
    }
    f3.close();

    ofstream f4("part2_error.csv");
    f4 << "c,h,error\n";
    for (double c = 0.2; c <= 10.0; c += 0.1)
    {
        double h_real = c * n_real;
        double error_real = calculate_E(X_real, Y_real, h_real);
        f4 << c << "," << h_real << "," << error_real << "\n";
    }
    f4.close();

    return 0;
}
