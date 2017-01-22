# CCD

* This is the coordinate descent algorithm for matrix factorizaiton.

### Usage
* g++ -L -lpthread -fopenmp  -o ccd *h *cpp
* ./ccd -n 4 ./toy-example/


### Time Complexity
* The time complexity is  O(|nnz|k^2), where |nnz| denotes the number of non-zero value.

### Reference
* Some codes are from http://www.cs.utexas.edu/~rofuyu/libpmf/. Thanks.
