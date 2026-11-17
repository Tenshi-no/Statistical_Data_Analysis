#include <iostream> //-> ��� �������������� ����
#include <cmath> //-> ��� ������������� ���. �������
#include <random> //-> ��� ���������������� ��������
#include <vector> //-> ������������ �������
#include <iomanip> //-> ��� ��������� ������

using namespace std;
//����� ��������� ��������� ����� (static ����� �� ������������ ��������� ������ ���)
static random_device rd;
static mt19937 gen(rd());
normal_distribution<double> dist(0.0, 1);
// ������� ��������� ��������� ������ h
double noise(double sigma = 0.5)
{
    return dist(gen)*sigma;
    return 0;
}

void run_pfe(int n, double s, double delta_u)
{
    int numeric_exp = pow(2, n); // ���������� ������
    vector<vector<double>> matrix_x(numeric_exp, vector<double>(n));
    vector<vector<double>> matrix_u(numeric_exp, vector<double>(n));
    vector<double> vector_y(numeric_exp);

    // ������������� ������� � ���������� ������
    for (int i = 0; i < numeric_exp; ++i)
    {
        double sum_u_squared = 0.0;

        // ���������� ����� �������� ��� �������� ����� � ������� ������� �������
        for (int j = 0; j < n; ++j)
        {
            // ��� ����� ����� ��� j-�� ������� ����� 2^j
            int step = pow(2, j);
            // ���� ������� �� ������� ������ ����� �������� - ������ -1, ���� ������: 1
            if ((i / step) % 2 == 1)
            {
                matrix_x[i][j] = -1.0;
            }
            else
            {
                matrix_x[i][j] = 1.0;
            }

            // �������� �� ������������ �������� � ��������
            matrix_u[i][j] = s + matrix_x[i][j] * delta_u;
            sum_u_squared = sum_u_squared + pow(matrix_u[i][j], 2);
        }
        vector_y[i] = sum_u_squared + noise(0.5);
    }

    // 1. ������ ������������� ������������ ������ (BETA)
    double beta_0 = 0.0;
    for (int i = 0; i < numeric_exp; ++i)
    {
        beta_0 += vector_y[i];
    }
    beta_0 = beta_0 / numeric_exp;

    vector<double> beta(n, 0.0);
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < numeric_exp; ++i)
        {
            beta[j] += matrix_x[i][j] * vector_y[i];
        }
        beta[j] = beta[j] / numeric_exp;
    }

    // 2. �������� � ����������� ���������� (ALPHA)
    double alpha_0 = beta_0;
    vector<double> alpha(n, 0.0);
    for (int j = 0; j < n; ++j)
    {
        alpha[j] = beta[j] / delta_u;
    }

    // ����� ����������� � �������
    cout << "\nPlanning matrix and responses:\n";
    cout << "N\t";
    for (int j = 0; j < n; ++j) cout << "X" << j+1 << "\t";
    for (int j = 0; j < n; ++j) cout << "U" << j+1 << "\t";
    cout << "Y\n";

    for (int i = 0; i < numeric_exp; ++i)
    {
        cout << i + 1 << "\t";
        for (int j = 0; j < n; ++j) cout << (matrix_x[i][j] > 0 ? "+1\t" : "-1\t");
        for (int j = 0; j < n; ++j) cout << matrix_u[i][j] << "\t";
        cout << vector_y[i] << "\n";
    }

    cout << "\nCalculated model parameters:\n";
    cout << "Code (beta):\n";
    cout << "  beta_0 = " << beta_0 << "\n";
    for (int j = 0; j < n; ++j)
    {
        cout << "  beta_" << j + 1 << " = " << beta[j] << "\n";
    }

    cout << "Natural (alpha):\n";
    cout << "  alpha_0  = " << alpha_0 << "\n";
    for (int j = 0; j < n; ++j)
    {
        cout << "  alpha_" << j + 1 << "  = " << alpha[j] << "\n";
    }
    cout << endl;
}


void run_drob_replik(int n, int k, double s, double delta_u)
{
    // ���������� ������ ����� ��������� 2^(n-k)
    int numeric_exp = pow(2, n - k);
    vector<vector<double>> matrix_x(numeric_exp, vector<double>(n));
    vector<vector<double>> matrix_u(numeric_exp, vector<double>(n));
    vector<double> vector_y(numeric_exp);

    for (int i = 0; i < numeric_exp; ++i)
    {
        double sum_u_squared = 0.0;
        // ��������� ������� ������� (-1 � 1)
        for (int j = 0; j < n - k; ++j)
        {
            int step = pow(2, j);
            if ((i / step) % 2 == 1)
            {
                matrix_x[i][j] = -1.0;
            }
            else
            {
                matrix_x[i][j] = 1.0;
            }
        }

        // ����������� ����������� ������� ����� ������������ �����������
        if (n == 4 && k == 1)
        {
            matrix_x[i][3] = matrix_x[i][0] * matrix_x[i][1] * matrix_x[i][2];
        }
        else if (n == 6 && k == 2)
        {
            // ������ ������������ ����������� ������ �� ����������� ������� �������� (������� 0, 1, 2, 3)
            matrix_x[i][4] = matrix_x[i][0] * matrix_x[i][1] * matrix_x[i][2]; // X5 = X1 * X2 * X3
            matrix_x[i][5] = matrix_x[i][0] * matrix_x[i][2] * matrix_x[i][3]; // X6 = X1 * X3 * X4
        }

        for (int j = 0; j < n; ++j)
        {
            // �������� �� ������������ �������� � ��������
            matrix_u[i][j] = s + matrix_x[i][j] * delta_u;
            sum_u_squared = sum_u_squared + matrix_u[i][j] * matrix_u[i][j];
        }
        vector_y[i] = sum_u_squared + noise(0.5);
    }

    // 1. ������ ������������� ������������ ������ (BETA)
    double beta_0 = 0.0;
    for (int i = 0; i < numeric_exp; ++i)
    {
        beta_0 += vector_y[i];
    }
    beta_0 = beta_0 / numeric_exp;

    vector<double> beta(n, 0.0);
    cout << "Coef model beta:\n";
    cout << "  beta_0 = " << beta_0 << "\n";
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < numeric_exp; ++i)
        {
            beta[j] += matrix_x[i][j] * vector_y[i];
        }
        beta[j] /= numeric_exp;
        cout << "  beta_" << j + 1 << " = " << beta[j] << "\n";
    }
    cout << endl;

    // 2. �������� � ����������� ���������� (ALPHA)
    double alpha_0 = beta_0;
    vector<double> alpha(n, 0.0);
    for (int j = 0; j < n; ++j) {
        alpha[j] = beta[j] / delta_u;
    }

    cout << "Coef model alpha:\n";
    cout << "  alpha_0 = " << alpha_0 << "\n";
    for (int j = 0; j < n; ++j) {
        cout << "  alpha_" << j + 1 << " = " << alpha[j] << "\n";
    }
    cout << endl;
}


void second_order_planning(double s, double delta_u)
{
   int n = 2;
   int n_core = pow(2, n);
   int n_star = 2 * n;
   int n_center = 1;
   double alpha_plan = 1.000; // �������� ��������� ����� (���������� alpha_plan, ����� �� ������ � ����. ������)

   int numeric_exp = n_core + n_star + n_center;
   vector<vector<double>> matrix_x(numeric_exp, vector<double>(n));
   vector<vector<double>> matrix_u(numeric_exp, vector<double>(n));
   vector<double> vector_y(numeric_exp);

   // 1. ���������� ���� �����
   matrix_x[0][0] =  1.0; matrix_x[0][1] =  1.0;
   matrix_x[1][0] = -1.0; matrix_x[1][1] =  1.0;
   matrix_x[2][0] =  1.0; matrix_x[2][1] = -1.0;
   matrix_x[3][0] = -1.0; matrix_x[3][1] = -1.0;

   // 2. �������� �����
   matrix_x[4][0] =  alpha_plan; matrix_x[4][1] =  0.0;
   matrix_x[5][0] = -alpha_plan; matrix_x[5][1] =  0.0;
   matrix_x[6][0] =  0.0;        matrix_x[6][1] =  alpha_plan;
   matrix_x[7][0] =  0.0;        matrix_x[7][1] = -alpha_plan;

   // 3. ����������� �����
   matrix_x[8][0] =  0.0;        matrix_x[8][1] =  0.0;

   // 4. ������ ����������� ��������� U � ������� Y
   for (int i = 0; i < numeric_exp; ++i)
   {
       double sum_u_squared = 0.0;
       for (int j = 0; j < n; ++j)
       {
           matrix_u[i][j] = s + matrix_x[i][j];
           sum_u_squared += pow(matrix_u[i][j], 2);
       }
       vector_y[i] = sum_u_squared + noise(0.5);
   }

   // 5. ���������� ������� ��������� ��� ������������� ���������� x' � ������ 3.10
   double sum_x1_sq = 0.0, sum_x2_sq = 0.0;
   for (int i = 0; i < numeric_exp; ++i)
   {
       sum_x1_sq += pow(matrix_x[i][0], 2);
       sum_x2_sq += pow(matrix_x[i][1], 2);
   }
   double x1_sq_mean = sum_x1_sq / numeric_exp;
   double x2_sq_mean = sum_x2_sq / numeric_exp;

   // 6. ������ ������������ ������������� BETA
   double beta_0 = 0.0, beta_1 = 0.0, beta_2 = 0.0, beta_12 = 0.0;
   double div_linear = 0.0;

   for (int i = 0; i < numeric_exp; ++i)
   {
       beta_0 += vector_y[i];
       beta_1 += matrix_x[i][0] * vector_y[i];
       beta_2 += matrix_x[i][1] * vector_y[i];
       div_linear += pow(matrix_x[i][0], 2);
   }
   beta_0 /= numeric_exp;
   beta_1 /= div_linear;
   beta_2 /= div_linear;

   // ������ ������� �������������� (������ �� ���� �������, �����������)
   for (int i = 0; i < numeric_exp; ++i)
   {
       beta_12 += matrix_x[i][0] * matrix_x[i][1] * vector_y[i];
   }
   beta_12 /= n_core; // ����������� (x1*x2, x1*x2) ����� ����� ����� ���� n_core = 4

   // ������������ ������������ ������� �� �������������������� �������� x'
   double beta_11 = 0.0, beta_22 = 0.0, sum_quad = 0.0;
   for (int i = 0; i < numeric_exp; ++i)
   {
       double x1_ep = pow(matrix_x[i][0], 2) - x1_sq_mean;
       double x2_ep = pow(matrix_x[i][1], 2) - x2_sq_mean;

       beta_11 += x1_ep * vector_y[i];
       beta_22 += x2_ep * vector_y[i];
       sum_quad += pow(x1_ep, 2);
   }
   beta_11 /= sum_quad;
   beta_22 /= sum_quad;

   // 7. �������� � ����������� ������������ ALPHA
   double alpha_11 = beta_11 / delta_u * delta_u;
   double alpha_22 = beta_22 / delta_u * delta_u;
   double alpha_12 = beta_12 / delta_u * delta_u;

   double alpha_1 = beta_1 / delta_u;
   double alpha_2 = beta_2 / delta_u;

   // �������������� ���������� ����� �������� ������������ ������ � ������ �����
   double beta_0_true = beta_0 - beta_11 * x1_sq_mean - beta_22 * x2_sq_mean;
   double alpha_0 = beta_0_true;

   // ����� �����������
   cout << fixed << setprecision(4);
   cout << "Coef beta (coded):\n";
   cout << "  beta_0  = " << beta_0  << "\tbeta_1  = " << beta_1  << "\tbeta_2  = " << beta_2 << endl;
   cout << "  beta_12 = " << beta_12 << "\tbeta_11 = " << beta_11 << "\tbeta_22 = " << beta_22 << endl << endl;

   cout << "Coef alpha (natural):\n";
   cout << "  alpha_0  = " << alpha_0  << "\talpha_1  = " << alpha_1  << "\talpha_2  = " << alpha_2 << endl;
   cout << "  alpha_12 = " << alpha_12 << "\talpha_11 = " << alpha_11 << "\talpha_22 = " << alpha_22 << endl << endl;

   // ����� ����������� ������� ������������
   cout << "====================================================================================\n";
   cout << "n\tx0\tx1\tx2\tx1x2\tx1^2\tx2^2\tx1'\tx2'\t|\tY\n";
   cout << "====================================================================================\n";

   for (int i = 0; i < numeric_exp; ++i)
   {
       double x0 = 1.0;
       double x1 = matrix_x[i][0];
       double x2 = matrix_x[i][1];
       double x1x2 = x1 * x2;
       double x1_sq = pow(x1, 2);
       double x2_sq = pow(x2, 2);
       double x1_ep = x1_sq - x1_sq_mean;
       double x2_ep = x2_sq - x2_sq_mean;

       cout << i + 1 << "\t"
            << setw(4) << setprecision(1) << x0 << "\t"
            << setw(4) << x1 << "\t"
            << setw(4) << x2 << "\t"
            << setw(4) << x1x2 << "\t"
            << setw(4) << x1_sq << "\t"
            << setw(4) << x2_sq << "\t"
            << setw(5) << setprecision(2) << x1_ep << "\t"
            << setw(5) << x2_ep << "\t|\t"
            << setprecision(4) << vector_y[i] << "\n";
   }
   cout << "====================================================================================\n";
}


int main()
{
    double s = 4.0;
    double delta_u = 1.0;

    run_pfe(2, s, delta_u);
    run_pfe(2, s, delta_u);

    int k = 2;
    run_drob_replik(6,k,s,delta_u);

    second_order_planning(s, delta_u);

    return 0;
}
