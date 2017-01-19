#include <iostream>
#include "ccd.h"
using namespace std;

int main(int argc, char *argv[])
{

    CCD inst;

    inst.parseParameter(argc, argv);
    inst.dumpParameter();
    inst.run();

    cout<<"hello"<<endl;

    return 0;
}
