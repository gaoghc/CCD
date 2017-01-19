#ifndef CCD_H
#define CCD_H

#include "util.h"

class CCD
{
public:
    CCD();

    void parseParameter(int argc, char **argv);
    void dumpParameter();


    void updateW();
    void updateH();


    double calObj();

    void run();


private:
     void exit_with_help();
     void init();

public:
    int k;
    int threads;
    int maxiter, maxinneriter;
    double lambda;
    double rho;
    double eta0, betaup, betadown;  // learning rate parameters used in DSGD
    int lrate_method, num_blocks;
    int do_predict, verbose;
    bool with_weights;

    smat_t R, Rt;
    mat_t W,H;

    char input_file_name[1024];
    char model_file_name[1024];

    double loss;
    bool done;
};

#endif // CCD_H
