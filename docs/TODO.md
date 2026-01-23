# Getting Started - Neural Network in C

## Prerequisites

- **C Compiler**: GCC, Clang, or MSVC
- **CMake**: Version 3.12 or higher
- **Git**: For version control
- **GitLab**: For CI/CD pipeline (optional)

## Quick Start

### 1. Build the Project

#### On Linux/macOS:
```bash
cd nn
mkdir build
cd build
cmake ..
make
```

#### On Windows (with Visual Studio):
```bash
cd nn
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### 2. Run Tests

```bash
# From build directory
ctest --output-on-failure
```

### 3. Run Examples

```bash
# From build directory
./examples/simple_network
```

## Development Workflow

### Step 1: Implement Core Structures

Start by implementing the basic data structures in [neuralnet.h](../include/neuralnet.h):

- Matrix structure for linear algebra operations
- Neural network structure to hold layers and parameters
- Memory management functions

### Step 2: Implement Activation Functions

Add activation functions in [neuralnet.c](../src/neuralnet.c):

- Sigmoid
- ReLU
- Tanh
- Softmax

### Step 3: Implement Forward Propagation

Create the forward pass algorithm:
1. Input → Hidden layers → Output
2. Apply activation functions at each layer

### Step 4: Implement Backpropagation

Create the backward pass:
1. Calculate output error
2. Propagate error backward through layers
3. Update weights and biases

### Step 5: Add Training Loop

Implement the training algorithm:
1. Initialize weights
2. For each epoch:
   - Forward propagation
   - Calculate loss
   - Backward propagation
   - Update weights

### Step 6: Test Your Implementation

Write tests in [test_basic.c](../tests/test_basic.c):
- Test matrix operations
- Test activation functions
- Test forward propagation
- Test backpropagation
- Test full training loop

### Step 7: Create Examples

Build practical examples in `examples/`:
- XOR problem
- Simple classification
- Regression task

## GitLab CI/CD Setup

### 1. Push to GitLab

```bash
git init
git add .
git commit -m "Initial neural network project structure"
git remote add origin <your-gitlab-repo-url>
git push -u origin main
```

### 2. Pipeline Execution

The pipeline will automatically:
- Build the project on each commit
- Run all tests
- Create deployment artifacts on main branch

### 3. View Pipeline

- Go to your GitLab repository
- Navigate to CI/CD → Pipelines
- View build logs and test results

## Debugging Tips

### Enable Debug Build

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Use Valgrind (Linux)

```bash
valgrind --leak-check=full ./tests/test_basic
```

### Use GDB

```bash
gdb ./tests/test_basic
```

## Next Steps

1. Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed implementation steps
2. Refer to [API_REFERENCE.md](API_REFERENCE.md) for function signatures
3. Check [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for organization details

## Resources

- **Deep Learning Book**: https://www.deeplearningbook.org/
- **Neural Networks from Scratch**: https://nnfs.io/
- **CMake Tutorial**: https://cmake.org/cmake/help/latest/guide/tutorial/
- **GitLab CI/CD Docs**: https://docs.gitlab.com/ee/ci/
