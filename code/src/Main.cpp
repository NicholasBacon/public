#include <vector>
#include <iostream>
#include <iomanip>

#include <chrono>
#include "mpi.h"
#include <math.h>


#include <list>
#include <cstring>    /* memset & co. */
#include <ctime>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <iostream>
#include <numeric>

#include<unistd.h>


int n = 10;
int startbytes =1;
int endbytes = 150000;

bool server = false;
int testcount = 100;
//int NG_START_PACKET_SIZE = 1028;
//int max_datasize = 10000000 * 2;
int maxbuffercount = 100000000; // 100M floats


bool selfPack;
MPI_Datatype datatype = MPI_FLOAT;
int total_size;

int rank, num_procs;

/* From stackoverflow */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template<typename T>
T variance(const std::list<T> &vec) {
    const size_t sz = vec.size();
    if (sz <= 1) {
        return 0.0;
    }

    // Calculate the mean
    const T mean = std::accumulate(vec.begin(), vec.end(), 0.0) / sz;

    // Now calculate the variance
    auto variance_func = [&mean, &sz](T accumulator, const T &val) {
        return accumulator + ((val - mean) * (val - mean) / (sz - 1));
    };

    return std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
}

void pingpong(void *buffer , int i) {

    MPI_Status status;
    int size_s;
//abslute modle times
//retal error.
    MPI_Type_size(datatype, &size_s);

    for (int j = 0; j < testcount; ++j) {
        if (rank == 0) {
            MPI_Send(buffer, i
                    , MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buffer, i
                    , MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);
        }
        if (rank == 1) {
            MPI_Recv(buffer, i, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Send(buffer, i, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }


        for (int j = 0; j < 10; ++j) {
            if (rank == 0) {
                MPI_Send(buffer, i, datatype, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, i, datatype, 1, 0, MPI_COMM_WORLD, &status);
            }
            if (rank == 1) {
                MPI_Recv(buffer, i, datatype, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Send(buffer, i, datatype, 0, 0, MPI_COMM_WORLD);
            }
        }

        std::list<double> times0, times1;

    int x = 3;

        for (int k = 0; k < 10; ++k) {

            double t1 = MPI_Wtime();
//            for (int j = 0; j < testcount; ++j) {
//                if (rank == 0) {
//                    MPI_Send(((int *) buffer) + j + k * testcount, i, datatype, 1, 0, MPI_COMM_WORLD);
//                    MPI_Recv(((int *) buffer) + j + 1 + k * testcount, i, datatype, 1, 0, MPI_COMM_WORLD, &status);
//                }
//                if (rank == 1) {
//                    MPI_Recv(((int *) buffer) + j + 1 + k * testcount, i, datatype, 0, 0, MPI_COMM_WORLD, &status);
//                    MPI_Send(((int *) buffer) + j + k * testcount, i, datatype, 0, 0, MPI_COMM_WORLD);
//                }
//            }

            double tfinal2 = (MPI_Wtime() - t1) / ( testcount);
            times1.push_back(tfinal2);



            double t0 = MPI_Wtime();
            for (int j = 0; j < testcount; ++j) {
                if (rank == 0) {
                    MPI_Send(buffer, i, datatype, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(buffer, i, datatype, 1, 0, MPI_COMM_WORLD, &status);
                }
                if (rank == 1) {
                    MPI_Recv(buffer, i, datatype, 0, 0, MPI_COMM_WORLD, &status);
                    MPI_Send(buffer, i, datatype, 0, 0, MPI_COMM_WORLD);
                }
            }
            double tfinal1 = (MPI_Wtime() - t0) / (testcount);


            times0.push_back(tfinal1);


        }

        times0.sort();
        times1.sort();

        int count = 0;
        double x00 = 0;
        double x50 = 0;
        double x90 = 0;
        for (const auto &item: times0) {
            if (count == 0) {
                x00 = item;
            }
            if (count == 5) {
                x50 = item;
            }
            if (count == 9) {
                x90 = item;
            }

            count++;

        }
        count = 0;
        double x01 = 0;
        double x51 = 0;
        double x91 = 0;

        for (const auto &item: times1) {
            if (count == 0) {
                x01 = item;
            }
            if (count == 5) {
                x51 = item;
            }
            if (count == 9) {
                x91 = item;
            }
            count++;

        }


        double mean1 = std::accumulate(times1.begin(), times1.end(), 0.0) / times1.size();
        double mean0 = std::accumulate(times0.begin(), times0.end(), 0.0) / times0.size();


        printf("%i,%15.9f,%15.9f,%15.9f,%15.9f,%15.9f,%15.9f,%15.9f,%15.9f\n", i * size_s, mean1, x01, x51, x91,mean0,x00, x50, x90);
        fflush(stdout);


}

#if 0
void pingpong(void *buffer) {

    MPI_Status status;
    int size_s;

    MPI_Type_size(datatype, &size_s);
    int i = 55;


    for (int j = 0; j < testcount; ++j) {
        if (rank == 0) {
            MPI_Send(buffer, i
                    , MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(buffer, i
                    , MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);
        }
        if (rank == 1) {
            MPI_Recv(buffer, i, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Send(buffer, i, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }


    fflush(stdout);

    int x = 3;
    for (i =  start/size_s; i < 150000; i = i * 2) {

        for (int j = 0; j < 10; ++j) {
            if (rank == 0) {
                MPI_Send(buffer, i, datatype, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buffer, i, datatype, 1, 0, MPI_COMM_WORLD, &status);
            }
            if (rank == 1) {
                MPI_Recv(buffer, i, datatype, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Send(buffer, i, datatype, 0, 0, MPI_COMM_WORLD);
            }
        }


        std::list<double> times0, times1;


        for (int k = 0; k < 10; ++k) {

            double t1 = MPI_Wtime();
//            for (int j = 0; j < testcount; ++j) {
//                if (rank == 0) {
//                    MPI_Send(((int *) buffer) + j + k * testcount, i, datatype, 1, 0, MPI_COMM_WORLD);
//                    MPI_Recv(((int *) buffer) + j + 1 + k * testcount, i, datatype, 1, 0, MPI_COMM_WORLD, &status);
//                }
//                if (rank == 1) {
//                    MPI_Recv(((int *) buffer) + j + 1 + k * testcount, i, datatype, 0, 0, MPI_COMM_WORLD, &status);
//                    MPI_Send(((int *) buffer) + j + k * testcount, i, datatype, 0, 0, MPI_COMM_WORLD);
//                }
//            }

            double tfinal2 = (MPI_Wtime() - t1) / ( testcount);
            times1.push_back(tfinal2);



            double t0 = MPI_Wtime();
            for (int j = 0; j < testcount; ++j) {
                if (rank == 0) {
                    MPI_Send(buffer, i, datatype, 1, 0, MPI_COMM_WORLD);
                    MPI_Recv(buffer, i, datatype, 1, 0, MPI_COMM_WORLD, &status);
                }
                if (rank == 1) {
                    MPI_Recv(buffer, i, datatype, 0, 0, MPI_COMM_WORLD, &status);
                    MPI_Send(buffer, i, datatype, 0, 0, MPI_COMM_WORLD);
                }
            }
            double tfinal1 = (MPI_Wtime() - t0) / (testcount);


            times0.push_back(tfinal1);


        }

        times0.sort();
        times1.sort();

        int count = 0;
        double x00 = 0;
        double x50 = 0;
        double x90 = 0;
        for (const auto &item: times0) {
            if (count == 0) {
                x00 = item;
            }
            if (count == 5) {
                x50 = item;
            }
            if (count == 9) {
                x90 = item;
            }

            count++;

        }
        count = 0;
        double x01 = 0;
        double x51 = 0;
        double x91 = 0;

        for (const auto &item: times1) {
            if (count == 0) {
                x01 = item;
            }
            if (count == 5) {
                x51 = item;
            }
            if (count == 9) {
                x91 = item;
            }
            count++;

        }


        double mean1 = std::accumulate(times1.begin(), times1.end(), 0.0) / times1.size();
        double mean0 = std::accumulate(times0.begin(), times0.end(), 0.0) / times0.size();


        printf("%i,%15.9f,%15.9f,%15.9f,%15.9f,%15.9f,%15.9f,%15.9f,%15.9f\n", i * size_s, mean1, x01, x51, x91,mean0,x00, x50, x90);
        fflush(stdout);
    }

}
#endif

static void *mpi_cuda_malloc(size_t count) {
    void *d_data1;
    cudaMalloc((void **) &d_data1, count * sizeof(float));
//    cudaMalloc((void **) &d_data0, data_size * sizeof(float));

    float *h_data = (float *) malloc(count * sizeof(float));

    for (int i = 0; i < count; ++i) {
        h_data[i] = i * 1.0f;
    }
//    cudaMemcpy(h_data, d_data0, data_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("%p\n",d_data1);
    fflush(stdout);

    // Copy initialized host data to GPU buffer.
    gpuErrchk(cudaMemcpy(d_data1, h_data, count * sizeof(float), cudaMemcpyHostToDevice));
    free(h_data);
    return d_data1;
}


struct ng_loggp_tests_val;

int loggp_do_benchmarks();

int loggp_prepare_benchmarks();

static int be_a_server(void *buffer, int size, int n, double d, char o_r,
                       MPI_Datatype dt_r, MPI_Datatype dt_s, int size_r,
                       int size_s);

static int be_a_client(void *buffer, int size, struct ng_loggp_tests_val *values,
                       MPI_Datatype dt_r, MPI_Datatype dt_s, int size_r,
                       int size_s);

/* a is object1 that holds all the tests for a specific PRTT(n,d,s)
 * in microseconds */
struct ng_loggp_tests_val {
    double *data; /* data array */
    int n, s; /* n,s values s in bytes*/
    double d; /* d (delay time) in usec */
    int testc, itestc; /* maximal and actual testcount */
    char o_r; /* should we measure o_r or not? - if yes, measure o_r at server and communicate it to client */
    void (*constructor)(struct ng_loggp_tests_val *a, int testc, int n, double d, int s);

    void (*destructor)(struct ng_loggp_tests_val *a);

    void (*addval)(struct ng_loggp_tests_val *a, double val);

    double (*getmed)(struct ng_loggp_tests_val *a);
};

/* object function definitions :) */

/* constructor for ng_loggp_tests_val class */
static void ng_loggp_tests_val_constr(struct ng_loggp_tests_val *a, int testc, int n, double d, int s) {
    a->n = n;
    a->d = d;
    a->s = s;
    a->testc = testc;
    a->o_r = 0;
    a->itestc = 0; /* number of tests in array */
    a->data = (double *) malloc(testc * sizeof(double));
    {
        int itestc;
        for (itestc = 0; itestc < testc; itestc++) {
            a->data[itestc] = 0.0;
        }
    }

}

/* destructor for ng_loggp_tests_val */
static void ng_loggp_tests_val_destr(struct ng_loggp_tests_val *a) {
    if (a->data != NULL) free(a->data);
    a->data = NULL;
}

/* add a measurement value */
static void ng_loggp_tests_val_addval(struct ng_loggp_tests_val *a, double val) {

    if (a->itestc < a->testc) {
        a->data[(a->itestc)++] = val;
    } else {
//    ng_error("too many tests (a should not happen!)\n");
    }
}

/* get median of all mesurements  */
static double ng_loggp_tests_val_getmed(struct ng_loggp_tests_val *a) {

    /* bubble-sort data */
    int x, y;
    double holder;

    for (x = 0; x < a->itestc; x++)
        for (y = 0; y < a->itestc - 1; y++)
            if (a->data[y] > a->data[y + 1]) {
                holder = a->data[y + 1];
                a->data[y + 1] = a->data[y];
                a->data[y] = holder;
            }

    /* return median */
    y = (a->itestc + 1) / 2;
    return a->data[y];
}

/* a object holds a full PRTT(n,d,s) for a fixed n and d */
typedef struct {
    int size;
    double value;
} t_sizevalue;

struct ng_loggp_prtt_val {
    t_sizevalue *data; /* data array */
    int n; /* n values */
    double d; /* d value in microseconds */
    int elems; /* # of elements in the data array */
    double a, b, lsquares; /* curve parameters, y=ax+b */
    void (*constructor)(struct ng_loggp_prtt_val *a, int n, double d);

    void (*destructor)(struct ng_loggp_prtt_val *a);

    void (*addval)(struct ng_loggp_prtt_val *a, int s, double val);

    void (*getfit)(struct ng_loggp_prtt_val *a, int lower, int upper);

    void (*remove)(struct ng_loggp_prtt_val *a, int item);
};

/* constructor for ng_loggp_prtt_val class */
static void ng_loggp_prtt_val_constr(struct ng_loggp_prtt_val *a, int n, double d) {
    a->n = n;
    a->d = d;
    a->elems = 0;
    a->data = NULL;
}

/* destructor for ng_loggp_prtt_val class */
static void ng_loggp_prtt_val_destr(struct ng_loggp_prtt_val *a) {
    if (a->data != NULL) free(a->data);
    a->data = NULL;
}

/* calculate the parameters for y = ax + b in the interval [lower,upper] elements
 * if lower == upper == 0 -> fit all values */
static void ng_loggp_prtt_getfit(struct ng_loggp_prtt_val *a, int lower, int upper) {
    long double
            x_mean = 0,
            y_mean = 0,
            h = 0, j = 0;
    int iterator;
    int count = upper - lower;

    /* solve the linear least squares problem directly, see
     * http://de.wikipedia.org/wiki/Kleinste-Quadrate-Methode (sorry, it's
     * missing in the english variant) for details.
     */
    for (iterator = lower; iterator < upper; iterator++) {
//        printf("fit: %i - %f (%Lf, %Lf)\n", a->data[iterator].size, a->data[iterator].value, x_mean, y_mean);
        x_mean += a->data[iterator].size;
        y_mean += a->data[iterator].value;
    }
    x_mean /= count;
    y_mean /= count;
    for (iterator = lower; iterator < upper; iterator++) {
        h += (a->data[iterator].size - x_mean) * (a->data[iterator].value - y_mean);
        j += (a->data[iterator].size - x_mean) * (a->data[iterator].size - x_mean);
    }
    a->a = h / j;
    a->b = y_mean - a->a * x_mean;
    //printf("params: %lf, %lf (%i) (%Lf, %Lf, %Lf, %Lf)\n", a->a, a->b, count, x_mean, y_mean, h, j);

    /* calculate the least squares difference for a fixed msg-size
     * (x-axis) */
    a->lsquares = 0;
    for (iterator = lower; iterator < upper; iterator++) {
        double sq;

        sq = a->a * a->data[iterator].size + a->b - a->data[iterator].value;
        a->lsquares += sq * sq;
    }
    a->lsquares /= (count - 2);
    a->lsquares = sqrt(a->lsquares);
}

/* addval for ng_loggp_prtt_val class */
static void ng_loggp_prtt_addval(struct ng_loggp_prtt_val *a, int s, double val) {
    (a->elems)++;
    a->data = static_cast<t_sizevalue *>(realloc(a->data, a->elems * sizeof(t_sizevalue)));
    assert(a->data != NULL);
    a->data[a->elems - 1].size = s;
    a->data[a->elems - 1].value = val;
}

/* remove for ng_loggp_prtt_val class */
static void ng_loggp_prtt_remove(struct ng_loggp_prtt_val *a, int item) {
    t_sizevalue *tmp;
    int i, ind;

    (a->elems)--;
    tmp = static_cast<t_sizevalue *>(malloc(a->elems * sizeof(t_sizevalue)));
    assert(tmp != NULL);

    ind = 0;
    for (i = 0; i < a->elems + 1; i++) {
        if (i == item) continue;
        tmp[ind] = a->data[i];
        ind++;
    }
    free(a->data);
    a->data = tmp;
}


static void printparams(struct ng_loggp_prtt_val *gresults,
                        struct ng_loggp_prtt_val *results_1_0,
                        struct ng_loggp_prtt_val *results_n_d,
                        struct ng_loggp_prtt_val *results_n_0,
                        struct ng_loggp_prtt_val *results_o_r,
                        unsigned long data_size, FILE *out, int n, int lower, int upper) {
    double g, G, o_s, o_r, L;
    int ielem;


    g = gresults->b;
    G = gresults->a;

    ielem = results_n_d->elems - 1;
    o_s = (results_n_d->data[ielem].value - results_1_0->data[ielem].value) / (results_n_d->n - 1) -
          results_1_0->data[ielem].value /* =d */;
    o_r = results_o_r->data[ielem].value;
    L = results_1_0->data[0].value / 2;
    printf("L=%lf ", L);
    printf(" s=%i ", results_1_0->data[ielem].size);
    printf(" o_s=%lf ", o_s);
    printf(" o_r=%lf ", o_r);
    printf(" g=%lf ", g);
    printf(" G=%lf (%lf GiB/s)", G, 1 / G * 8.0 / 1024);
    printf(" lsqu(g,G)=%lf ", gresults->lsquares);
    if (results_n_d->d < g + G * data_size)
        printf("!!! d (%lf) is smaller than g+size*G (%lf) !!!\n", results_n_d->d, g + G * data_size);
    printf("\n");
}


/* a is the inner test loop from do_benchmarks ... we have to put
 * a in an extra function because we need to do the whole
 * benchmarkset more than once */
static int prtt_do_benchmarks(unsigned long data_size,
                              struct ng_loggp_tests_val *values, struct ng_loggp_prtt_val *results, char *buffer,
                              char o_r, MPI_Datatype dt_s, MPI_Datatype dt_c) {
    /** number of times to test the current datasize */
    unsigned long test_count = testcount;

    /** how long does the test run? */
    time_t test_time, cur_test_time;

    /** number of tests run */
    int test, ovr_tests, ovr_bytes;


    int size_s;
    int size_c;
    MPI_Type_size(dt_s, &size_s);
    MPI_Type_size(dt_c, &size_c);
    /* initialize tests object  */
    values->n = results->n;
    values->d = results->d;
    if (!server) {
        values->constructor(values, /* testcount = */ test_count, /* n =*/ values->n,
                /* d = */ values->d, /* s = */ 0);
    }
    values->o_r = o_r;

    test_time = 0;
    for (test = 0; test < test_count; test++) {
        if (server) {
            /* execute server mode function */
            be_a_server(buffer, data_size, values->n, (o_r ? 30000 : 0.0), o_r, dt_s, dt_c, size_s, size_c);
        } else {
            /* wait some time for the server to get ready */
            usleep(10);
            /* execute client mode function */
            be_a_client(buffer, data_size, values, dt_s, dt_c, size_s, size_c);
        }

    }
    if (!server) {
        double res;
        res = values->getmed(values);
        results->addval(results, data_size, res);
        values->destructor(values);
    }

    return 0;
}

/* the REAL benchmark loop - loops over all sizes for all three
 * benchmarks (PRTT(1,0,s), PRTT(n,0,s), PRTT(n,d,s) where d=PRTT(1,0,s)
 * and n is defined as const */
int loggp_do_benchmarks() {
    /** size of the buffer used for transmission tests */




    /** Output File */
    FILE *out = NULL;


    /* initialize the statistics */

    int res;

    /** to store the temporary results and define test parameters */
    struct ng_loggp_tests_val values = {
            .data= nullptr, /* data array */
            .n=0, /* n values */
            .s=0, /* n values */
            .d=0,/* d value in microseconds */
            .testc=0,
            .itestc=0, /* maximal and actual testcount */
            .o_r=0, /* should we measure o_r or not? - if yes, measure o_r at server and communicate it to client */
            .constructor = ng_loggp_tests_val_constr,
            .destructor = ng_loggp_tests_val_destr,
            .addval = ng_loggp_tests_val_addval,
            .getmed = ng_loggp_tests_val_getmed
    };



    /* stores the final results (median of tests) for n=1 and d=0 */
    struct ng_loggp_prtt_val results_1_0 = {
            .data= nullptr, /* data array */
            .n=0, /* n values */
            .d=0,/* d value in microseconds */
            .elems=0,/* # of elements in the data array */
            .a=0, .b=0, .lsquares=0,
            .constructor = ng_loggp_prtt_val_constr,
            .destructor = ng_loggp_prtt_val_destr,
            .addval = ng_loggp_prtt_addval,
            .getfit = ng_loggp_prtt_getfit,
            .remove = ng_loggp_prtt_remove,
    };

    /* stores the final results (median of tests) for arbitrary n and d=0 */
    struct ng_loggp_prtt_val results_n_0 = {
            .data= nullptr, /* data array */
            .n=0, /* n values */
            .d=0,/* d value in microseconds */
            .elems=0,/* # of elements in the data array */
            .a=0, .b=0, .lsquares=0,
            .constructor = ng_loggp_prtt_val_constr,
            .destructor = ng_loggp_prtt_val_destr,
            .addval = ng_loggp_prtt_addval,
            .getfit = ng_loggp_prtt_getfit,
            .remove = ng_loggp_prtt_remove,
    };


    /* stores the final results (median of tests) for arbitrary a and d */
    struct ng_loggp_prtt_val results_n_d = {
            .data= nullptr, /* data array */
            .n=0, /* n values */
            .d=0,/* d value in microseconds */
            .elems=0,/* # of elements in the data array */
            .a=0, .b=0, .lsquares=0,
            .constructor = ng_loggp_prtt_val_constr,
            .destructor = ng_loggp_prtt_val_destr,
            .addval = ng_loggp_prtt_addval,
            .getfit = ng_loggp_prtt_getfit,
            .remove = ng_loggp_prtt_remove,
    };

    /* stores the o_r results - it's a bit an abuse of a data structure
     * but it works conveniently */
    struct ng_loggp_prtt_val results_o_r = {
            .data= nullptr, /* data array */
            .n=0, /* n values */
            .d=0,/* d value in microseconds */
            .elems=0,/* # of elements in the data array */
            .a=0, .b=0, .lsquares=0,
            .constructor = ng_loggp_prtt_val_constr,
            .destructor = ng_loggp_prtt_val_destr,
            .addval = ng_loggp_prtt_addval,
            .getfit = ng_loggp_prtt_getfit,
            .remove = ng_loggp_prtt_remove,
    };

    /* a is just a temp. object to store
     * (PRTT(size,n,0)-PRTT(size,0,0))/(n-1) to fit g and G to a
     * values */
    struct ng_loggp_prtt_val gresults = {
            .data= nullptr, /* data array */
            .n=0, /* n values */
            .d=0,/* d value in microseconds */
            .elems=0,/* # of elements in the data array */
            .a=0, .b=0, .lsquares=0,
            .constructor = ng_loggp_prtt_val_constr,
            .destructor = ng_loggp_prtt_val_destr,
            .addval = ng_loggp_prtt_addval,
            .getfit = ng_loggp_prtt_getfit,
            .remove = ng_loggp_prtt_remove,
    };

    /* the famous n of PRTT(n,d,s) */


    /** currently tested packet size */
    unsigned long data_size;

    /** number of times to test the current datasize */
    unsigned long test_count = testcount;

    /* element of last protocol change */
    int lastchange = 0 /* last protocol change */;

    if (loggp_prepare_benchmarks()) return 1;


    char *buffer = (char *) mpi_cuda_malloc(maxbuffercount);

    MPI_Type_size(datatype, &total_size);


    results_1_0.constructor(&results_1_0, 1, 0);
    results_n_0.constructor(&results_n_0, n, 0);
    results_n_d.constructor(&results_n_d, n, results_n_d.d);
    results_o_r.constructor(&results_o_r, n, 0);
    gresults.constructor(&gresults, 1, 0);

    if (startbytes < total_size) startbytes = total_size;

    for (int i =  startbytes/total_size; i < endbytes/total_size; i = i * 2) {
        fflush(stdout);
        pingpong(buffer ,i);
        data_size = i * total_size;
        res = prtt_do_benchmarks(data_size, &values, &results_1_0, buffer, 0, datatype, datatype);
        res = prtt_do_benchmarks(data_size, &values, &results_n_0, buffer, 0, datatype, datatype);

        /* g needs to be fitted to: (PRTT(size,n,0)-PRTT(size,0,0))/(n-1) */
        if (!server) {

            /* add last measurement value to gresults */
            gresults.addval(&gresults,
                            results_n_0.data[results_1_0.elems - 1].size,
                            (results_n_0.data[results_1_0.elems - 1].value -
                             results_1_0.data[results_1_0.elems - 1].value) / (results_n_0.n - 1));
            //gresults.getfit(&gresults, lastchange, gresults.elems);
            /* take the PRTT(1,0,s) as delay - a is bigger than g+G*size :) */
            results_o_r.d = results_n_d.d = results_1_0.data[results_1_0.elems - 1].value;
        }

        /* results_o_r.d must be valid on client and server! */
        MPI_Bcast(&results_o_r.d, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        /* only set a once */
        res = prtt_do_benchmarks(data_size, &values, &results_n_d, buffer, 0, datatype, datatype);
        /* only set a once ,datatype,MPI_CHAR);
         ,MPI_CHAR,datatype);*/
        res = prtt_do_benchmarks(data_size, &values, &results_o_r, buffer, 1, datatype, datatype);

        /* evaluate the measurement results */


        if (!server) {


            int ielem;

            /* if lsquares-deviation of fit too high:
             * remove all extreme outliers from gresults -
             * an outlier is a value that is more than 2*lsquares(g,G) away from the fitted function
             */
            if (gresults.lsquares > 100) {
                for (ielem = 0; ielem < gresults.elems; ielem++) {
                    double lsquares = gresults.lsquares;

                    /* if a point is more than 2*lsquares above the line, it's
                     * probably an outlier */
                    if (gresults.data[ielem].value >
                        gresults.a * gresults.data[ielem].size + gresults.b + 2 * lsquares) {
                        double value = gresults.data[ielem].value;
                        unsigned int size = gresults.data[ielem].size;

                        /* remove value from elements */
                        gresults.remove(&gresults, ielem);
                        /* TODO: should we also remove it from other prtt_results ? */
                        gresults.getfit(&gresults, lastchange, gresults.elems);
                        printf("**** removed value %lf for size %u from gresults, lsquares was: %lf, new lsquares: %lf\n",
                               value, size, lsquares, gresults.lsquares);
                    }
                }
            }



            {
                const int x = 5; /* number of points to look ahead (+1) */
                //for(ielem = lastchange+3 /* we need 2 elements for a fit */; ielem<gresults.elems-x-1; ielem++) {
                if (lastchange + x /* look-ahead x elems */ + 2 /* we need 2 elems for fit */ <= gresults.elems) {
                    int ix /* runner */;
                    int flag = 1; /* are all bigger than f(x) + 2*lsquares ? -> 1 = yes */
                    double lsquares;
                    const double pfact = 2.0; /* wurschtel-factor */

                    ielem = gresults.elems - x;
                    /* get fit for lastchange up to current item */
                    //printf("getfit: %i, %i\n", lastchange, ielem);
                    gresults.getfit(&gresults, lastchange, ielem);
                    lsquares = gresults.lsquares;

                    /* look x elements ahead */
                    for (ix = ielem + 1; ix < ielem + x; ix++) {
                        gresults.getfit(&gresults, lastchange, ix);
                        /* only if all lsquares have at least doubled
                         * ... alles scheisse, wenn lsquares mal wirklich klein ist
                         * haben wir viele Protokollwechsel :-( */
                        if ((gresults.lsquares < pfact * lsquares) || isnan(lsquares) ||
                            (lsquares < 0.15) /* lower bound to prevent flapping */)
                            flag = 0;
                    }


                    /* if all x points are > f(x) + 2*lsquares, we have a protocol
                     * change, if only b < x points are larger, they are
                     * outliers and are removed in the next loop ... */
                    if (flag) {
                        /* we have a protocol change and the current element
                         * (ielem) is the last element in the old protocol */
                        printf("we detected a protocol change at %i bytes:\n", gresults.data[ielem].size);
                        gresults.getfit(&gresults, lastchange, ielem);
                        printparams(&gresults, &results_1_0, &results_n_d, &results_n_0, &results_o_r, data_size, out,
                                    n, lastchange, ielem);
                        lastchange = ielem + 1;
                        /* ok, we have now a new protocol beginning at ielem + 1,
                         * and we need to fit the new line to the next x elements */
                        //ielem += x; /* ATTENTION: we change the loop-runner here */
                        gresults.getfit(&gresults, lastchange, gresults.elems);
                    }
                } /* for(ielem = 0; ielem<gresults.elems ... */
                gresults.getfit(&gresults, lastchange, gresults.elems);
                printparams(&gresults, &results_1_0, &results_n_d, &results_n_0, &results_o_r, data_size, out, n,
                            lastchange, results_n_0.elems);
            }
        }
    }
    for (int j = 0; j < results_1_0.elems; ++j) {
        printf("n=%i,size-%i,1_0 %15.9f,n_0 %15.9f,n_d %15.9f,o_r %15.9f,gresults %15.9f;",n,
               results_1_0.data[j].size,
               results_n_0.data[j].value,
               results_n_d.data[j].value,
               results_o_r.data[j].value, gresults.data[j].value);
    }

    return 0;

}


int loggp_prepare_benchmarks() {
    /* only if we've got MPI */



    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    server = rank == 0;


    return 0;

}

static void my_wait(double d) {
    auto start = std::chrono::high_resolution_clock::now();
    while (d > std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - start).count());

}


int sendto(int dst, void *buffer, int size, MPI_Datatype datatype1) {
    MPI_Send(buffer, size, datatype1, dst, 13, MPI_COMM_WORLD);
    return 0;
}

int recvfrom(int src, void *buffer, int size, MPI_Datatype datatype1) {
    MPI_Recv(buffer, size, datatype1, src, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return 0;
}


static int
be_a_server(void *buffer, int size, int n, double d, char o_r /* measure o_r? */, MPI_Datatype dt_r, MPI_Datatype dt_s,
            int size_r,
            int size_s) {

    int in;
    const int partner = 1;
    if (o_r) {
        recvfrom(partner, buffer, size / size_r, dt_r);
        /* get start time */
        auto start = std::chrono::high_resolution_clock::now();
        /* Phase 1: receive data */
        for (in = 0; in < n - 1; in++) {
            my_wait(d);
            recvfrom(partner, buffer, size / size_r, dt_r);
        }
        auto stop = std::chrono::high_resolution_clock::now();
        sendto(partner, buffer, size / size_s, dt_s);
        double val;
        double duration = std::chrono::duration<double, std::micro>(stop - start).count();
        val = (duration - d * (n - 1)) / (n - 1);
        MPI_Send(&val, 1, MPI_DOUBLE, partner, 11, MPI_COMM_WORLD);


    } else {

        recvfrom(partner, buffer, size / size_r, dt_r);
        for (in = 0; in < n - 1; in++) {
            recvfrom(partner, buffer, size / size_r, dt_r);
        }
        sendto(partner, buffer, size / size_s, dt_s);


    }


    return 0;
}

static int
be_a_client(void *buffer, int size, struct ng_loggp_tests_val *values, MPI_Datatype dt_r, MPI_Datatype dt_s, int size_r,
            int size_s) {

    int in;
    const int partner = 0;
    auto start = std::chrono::high_resolution_clock::now();

    sendto(partner, buffer, size / size_s, dt_s);

    for (in = 0; in < values->n - 1; in++) {
        my_wait(values->d);
        sendto(partner, buffer, size / size_s, dt_s);

    }

    /* Phase 2: receive returned data */
    recvfrom(partner, buffer, size / size_r, dt_r);


    /* get after-receiving time */
    auto stop = std::chrono::high_resolution_clock::now();

    /* calculate results */


    double duration = std::chrono::duration<double, std::micro>(stop - start).count();


    if (values->o_r) { /* benchmark o_r */
        double val;
        MPI_Recv(&val, 1, MPI_DOUBLE, partner, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        values->addval(values, val);
    } else {
        values->addval(values, duration);
    }


    return 0;
}


std::vector<int> split(const std::string &s, char delim) {
    std::vector<int> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(atoi(item.c_str()));
    }

    return result;
}

MPI_Datatype make_datatype(int argc, char *argv[]) {

    int ret;
    MPI_Datatype oldType = MPI_FLOAT;
    std::vector<std::vector<int>> x;

    std::vector<int> v = split(argv[1], ',');
    x.insert(x.end(), v);
    MPI_Datatype type;
    ret = MPI_Type_vector(v[0], v[1], v[2], oldType, &type);
    assert(ret == MPI_SUCCESS);
    ret = MPI_Type_commit(&type);
    assert(ret == MPI_SUCCESS);

    return type;
}


int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);



//    MPI_Type_vector(3, 2, 4, MPI_FLOAT, &datatype);
    if (argc > 2) 
        startbytes= atoi(argv[2]);
    if (argc > 3) 
        endbytes= atoi(argv[3]);
    printf("testing from %d bytes to %d bytes\n",startbytes, endbytes);
    fflush(stdout);
    if (argc > 1) {
     datatype = make_datatype(argc, argv);
    }else{
       datatype =MPI_FLOAT;
    }
       
      
    loggp_prepare_benchmarks();

    loggp_do_benchmarks();

    MPI_Finalize();

    printf("\n");
    return 0;
}


