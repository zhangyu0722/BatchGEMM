#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<fstream>
#include<vector>
#include<algorithm>
#include<utility>

struct mytuple {
    int m;
    int n;
    int k;
};
using namespace std;

//#define random(x) (rand()%(x))
bool is64x64(int a, int b) { return (a % 64 == 0 && b % 64 == 0); }

bool is32x32(int a, int b) { return (a % 32 == 0 && b % 32 == 0 && !is64x64(a, b)); }

bool is16x16(int a, int b) { return (!is64x64(a, b) && !is32x32(a, b)); }

bool cmp_strategy(mytuple p1, mytuple p2) {
    if (is16x16(p1.m, p1.n)) return false;
    else if (is16x16(p2.m, p2.n)) return true;
    else if (is32x32(p1.m, p1.n)) return false;
    else if (is32x32(p2.m, p2.n)) return true;
    return false;
}

bool cmp_k(mytuple p1, mytuple p2) {
    return p1.k > p2.k;
}

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    int direction = atoi(argv[2]);
    fstream fs;
    fs.open("../data/data_MN_K_512_128");
    if (!fs.is_open()) {
        printf("Error opening input\n");
        exit(EXIT_FAILURE);
    }

    vector <mytuple> v;
    //read matrix config
    int K = 0;
    for (int i = 0; i < N; ++i) {
        int a, b, c;
        fs >> a >> b >> c;
        mytuple t;
        t.m = a;
        t.n = b;
        t.k = c;
        v.push_back(t);
    }
    if (direction == 1) {//combination
        stable_sort(v.begin(), v.end(), cmp_k);
        stable_sort(v.begin(), v.end(), cmp_strategy);
    }
    fs.close();
    fs.open("../data/data_MN_K_512_128_sort");
    for (int i = 0; i < N; ++i) {
        fs << v[i].m << "	" << v[i].n << "	" << v[i].k << endl;
    }

    return EXIT_SUCCESS;
}
