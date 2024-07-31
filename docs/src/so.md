# Interfacing with C code

The code in `MLNanoShaperRunner` can be compiled to run as a shared object and ca be interfaced as a shared library.

The first step is to compile the code.
```
julia --project MLNanoShaperRunner/build/build.jl
```

Once this is done, we have muliple directory in `MLNanoShaperRunner/build/lib`.
- `include` which contains the headers to be included in the C code.
- `lib` which contains the shared objects that need to be referenced by the code.
- `shared` which contains the artefacts necessary for the julia code. A copy of the `shared` directory must be included in the root project of the executable.


# Interface
First the Interface code define some structures.
```c
typedef struct {
  float x;
  float y;
  float z;
  float r;
} sphere;
```

and

```c
typedef struct {
  float x;
  float y;
  float z;
} point;
```



```c
int load_model(char *path);
```
Load the model `parameters` form a serialised training state at absolute path
`path`. Parameters: path - the path to a serialized NamedTyple containing the
parameters of the model Return value(int):
- 0: OK
- 1: file not found
- 2: file could not be deserialized properly
- 3: unknow error

```c
int load_atoms(sphere *start, int length);
```
Load the atoms into the julia model.
Start is a pointer to the start of the array of `sphere` and `length` is the
length of the array

## Return an error status:
- 0: OK
- 1: data could not be read
- 2: unknow error

```c
float eval_model(point *start,int length); 
```
evaluate the model at coordinates start[0],...,start[length -1]

# Example
```c
#include "MLNanoShaperRunner.h"
#include "julia_init.h"

int main(int argc,char *argv[]) {
  init_julia(argc, argv);
  load_model("/home/tristan/datasets/models/"
               "angular_dense_2Apf_epoch_10_16451353003083222301");
  sphere data[2]= {{0.,0.,0.,1.},{1.,0.,0.,1.}};
  load_atoms(data,2);
  point x[2] = {{0.,0.,1.},{1.,0.,0.}};
  eval_model(x,2);
  shutdown_julia(0);
  return 0;
}
```
