# Entropy Guided Optimizer

This repo implements a stochastic optimizer which uses Gaussian sampling, hierarchical clustering and adaptive refocus. The optimizer is designed for functions with complex landscapes where exploration-exploitation trade-offs are critical.

## Core Idea

1. It starts with an initial belief.
2. Samples from the initial belief.
3. Evaluates a function.
4. Check which samples are promising based on the threshold. If enough promising samples exist, trigger a refocus. The refocus adjusts the sampling distribution toward better regions. - Try refocus

   * Check which samples are promising based on the threshold. If enough promising samples exist, trigger a refocus. The refocus adjusts the sampling distribution toward better regions.
5. Finds the largest dominant sample region - macro clustering.
6. Within the most dominant region, finds the most local and strong region devoid of outliers - micro clustering.
7. Updates weights.
8. The weights are used to generate new beliefs.
9. The process continues like that until optimum is reached.

## Contents

1. `Makefile` - Build instructions and compilation targets
2. `Implementation.cpp` - Core Optimizer class and step functions
3. `Fobj.h` - Objective functions to be optimized
4. `multivariate.h` - Multivariate normal sampler and Cholesky decomposition utilities

## How to run

```bash
git clone entropy-guided-optimizer
cd entropy-guided-optimizer
# configure parameters in the implementation.cpp
make
./optimizer
```

## ALGORITHM DESIGN

## Implementation.cpp

### Headers

```cpp
#include <vector> // for dynamic arrays
#include <iostream> // couts, cerrs for debug prints etc
#include <array> // fixed size arrays, whose size is static and memory is contiguous, it's known at compile time, faster than vectors
#include <algorithm>//nth_element, move
#include <queue> // deque used for bfs to find neighbours
#include "multivariate.h" // multivariate function for sampling
#include "fobj.h" // objective function which is being solved for.
```

### Template for generic programming

* The template makes the class generic, it contains `size_t dim` & `size_t batch`. The goal of this was so that we could use both dim and batch to size our `arrays`. This would ensure the sizes were always known at compile time.

### Class Optimizer

* The class optimizer contains the following:

1. `struct HistoryEntry` - This contains the records of the iterations. It includes an array, a double and a bool.

2. `private`
   This one contains the following:

```cpp
int function; // function parameter
std::array<double,dim> mu0; // mean vector of sampling distribution
std::array<double, dim*dim> sigma0; // covariance matrix -> flattened describing the spread
double eta; // scaling factor when computing weighted mean - it influences weight computation
double tau_macro; // macro clustering threshold - influences the threshold for a macro cluster
double tau_micro; // min clustering threshold - influences selection for a micro cluster
double refocus_thres; // parameter to decide at what best_f to refocus on "promising samples"
int refocus_min_samples; // the minimum number of samples required in order for a refocus to happen
int random_seed; // seed random number for reproducibility
double refocus_change_rate; // refocus change rate, after every successful refocus, we change the refocus so that the next refocus it can increasingly focus on better regions
std::mt19937 rng;
```

The functions include the following which we will get into in a while:

```cpp
std::array<double,dim> mu0_generation() // generates the initial mean of sampling distributions, and returns a vector of zeros; we use an array as size is initially known and we want to avoid overhead that comes with dynamic STLs.

std::array<double, dim*dim> sigma_generation() // this one returns a diagonal covariance matrix 2D which is then flattened to a 1D array. This keeps everything in a contiguous memory line which is cache friendly and CPU friendly.

std::vector<bool> cluster_mask_macro(const std::array<double, batch*dim> &X_flat, double tau)  // This one is responsible for finding the most dominant sampling region devoid of outliers.

std::vector<bool> cluster_mask_micro(const std::vector<double> &Xm, double tau) // this one focuses on micro region, which is extracted from Xm specifically, Xm are samples produced specifically by the previous one

std::pair<bool,double> _try_refocus(std::array<double,batch*dim> &X_flat,std::array<double,batch> &fvals, double refocus_threshold,int refocus_min_samples ) // This one is responsible for the refocus step

std::pair<bool,double> step() // this is the optimization step which contains all these, as well as weight updates etc

std::vector<HistoryEntry> optimize(int iters) // The full optimization loop is ran which produces HistoryEntry for analysis
```

3. `public`

* This contains the only function the user is allowed to run which is:

```cpp
    void run(
        int function, 
        double eta_, 
        double tau_macro_,
        double tau_micro_,
        double refocus_thresh_,
        int refocus_min_samples_,
        int random_seed_,
        int iterations,
        double refocus_change_rate_
    )
```

## multivariate.h

### headers

```cpp
#pragma once
#include <random> // RNG
#include <array> // Arrays - allows compile time array sizing, no allocation overhead as compared to vectors
#include <cmath> // we use isfinite for numerical stability checks and sqrt
#include <limits> // numerical limits
#include <algorithm> // max
#include <iostream> // diagnostics cerr
```

### functions

```cpp
// limits extreme Gaussian values, normal distribution has infinite tails to prevent overflow - effectively prevents catastrophic values
inline double clamp_normal_value(double z, double limit = 8.0)

template <size_t dim>
// calculates a Cholesky with numerical safety inbuilt
bool safe_cholesky(const std::array<double, dim*dim>& Sigma,std::array<double, dim*dim>& L,double initial_jitter = 1e-12, int max_attempts = 8)

template <size_t dim, size_t batch>
// drives the sampling to create X_flat
std::array<double, batch*dim> multivariate(std::mt19937 &rng,const std::array<double, dim>& mu0,const std::array<double, dim*dim>& Sigma)
```

## Implementation Details

### implementation.cpp - Optimizer Class

1. `main()`

* This is the main program:

```cpp
Optimizer<10,3000> session; // allocates session to optimizer and we initialize it with a dim of 10 and sampling of 3000

// These are the defined parameters for our optimizer session
double eta = 0.1;
double tau_macro=0.3;
double tau_micro=0.2;
double refocus_thresh=-36.869;
int refocus_min_samples=100;
int random_seed=42;
int iterations=100;
double refocus_change_rate=0.5;

// we run this specifically to begin the optimization process, this takes us to run
session.run(
       26, // function
       eta, // eta 
       tau_macro, // tau macro
       tau_micro, // tau micro
       refocus_thresh, // refocus threshold
       refocus_min_samples, // refocus min samples
       random_seed, // random seed
       iterations, // iterations
       refocus_change_rate 
   )   ;

```

2. `session.run()`

* This is our public run section. It is the only public function we have in here.
* Its arguments are the parameters of the optimizer class.

  * Assigns the parameters required to run
  * initiates the optimization process by initializing mu0 and sigma0 once:

```cpp
mu0 = mu0_generation(); // this one calls mu generation
sigma0 = sigma_generation(); // this one calls sigma generation
```

* The mu0 and sigma0 generated are then saved into those variables which are available in our private parameters.

* Lets get into mu0 and sigma0 generation:
  *`mu0_generation` ->  it has no arguments and it's supposed to return an array of size `dim` which is the number of dimensions. The dim is always known at compile time because we used template generic programming to define it.

  * All elements in the mu0 are then converted to 0 in the array, this shapes our initial mu0 belief.
    *`sigma0_generation` -> it also has no arguments, what happens is it returns an array of size `dim * dim` which is a flat array. Originally it was a 2D but due to performance considerations and the need to keep it in stack memory, we opted for a 1D. The navigation is `r*dim + c`.
  * It returns a diagonal covariance matrix. For every column it sets 0 and if row number = column number, it sets 2.

* Next, in run, we have seeding:

```cpp
        // seed RNG once here (deterministic when desired)
        if (random_seed >= 0) rng.seed(static_cast<uint32_t>(random_seed));
        else rng.seed(std::random_device{}());
```

-Then we have the history production step which needs more details of our implementation:

```cpp
auto history = optimize(iterations);
```

* We use auto so that we ensure that once scope is finished, it will automatically go away.
* This step takes us to `optimize(iterations)` so lets get into it:

  *`optimize` is a private function on our Optimizer class. It returns a vector `std::vector<HistoryEntry>` which is responsible for recording all that the optimizer is doing in order to be able to return history which is then printed in our public run function.
  *to get into History struct which was mentioned here, I have chosen to get into the step function first as this function is called in our optimize function and ran and encompasses majority of the Optimizer process so buckle up!!

3. `std::pair<bool,double> step()`

* The step is where the majority of the optimization steps are actively being called and used to process the samples. Remember in run, we had already initialized mu0 and Sigma0 which means the initials exist. The reason we opted for producing the two in run instead of here is that if the generation was present here, then across every iteration, we would inevitably be getting erroneous results which would not be a continuity of the optimization.

* This step returns a pair which consists of a bool and a double, the bool basically signals whether this optimization step utilized `refocus` or not. The double is specifically the best function which is named `best_f`.

* Lets get into the steps in the step function:

- The mu0 and Sigma0 have already been generated so the next logical thing to do would be to get into the actual sampling which brings us to this line:

```cpp
std::array<double, batch*dim> X_flat = multivariate<dim, batch>(rng, mu0, sigma0);
```

* This line calls a multivariate function whose arguments include the rng, mu0 and sigma0.
* The goal of this is to produce samples which will then be stored into a Flat 1D array called `X_flat` of size `batch*dim`.
* The multivariate function is specifically available in our `multivariate.h` header which we briefly discussed earlier.
  Lets get into the process :

4. `multivariate.h`

* The first function is a `clamp_normal_value` which returns a double.

* We inline. Inline functions can eliminate overhead. It's also small enough to be inlined.

* The goal is this:

  * it has two arguments, double z which is the number being observed and double limit which is the hard limit.
  * It limits extreme Gaussian values. Normal distribution has infinite tails. So this is designed to prevent overflow which can lead to catastrophic values.
  * if z is NaN or inf, it replaces it with a safe value.
  * then the two ifs are for placing hard limits.
  * if z is safe then it does not change.

* The next function is `safe_cholesky` which returns a bool.

  * It is implemented with template programming which enables compiler optimizations in static arrays.

  * Its arguments are a flattened sigma which is referenced to avoid copying(`const std::array<double, dim*dim>& Sigma`)

  * A cholesky factor lower triangle which is also a flat array(`std::array<double, dim*dim>& L, // cholesky factor(lower triangle)`)

  * `initial_jitter` which is a tiny diagonal added to avoid singularity in matrices

  * `max_attempts` which are the max retries for increasing the jitter.

  * It detects invalid covariance entries. If they are present then it is impossible to do decomposition so it returns failure immediately.

  * The double jitter is set to be the initial jitter, which will grow if decomposition fails, to stably and reliably make the matrix stable.

  * The proceeding loop focuses on stabilizing the diagonal by increasing the jitter. This is because covariance matrices will often become near singular.

  * L is filled with zeros so that only the lower triangle will be written, this effectively reduces the computation by half.

  * The `bool ok = true` is the success flag to break loops if failure is detected, it is set to true initially.

  * The loop focuses on columns of lower triangle in all rows, the upper triangle ones are skipped.

  * Sum initializes the Cholesky accumulation using the covariance entry

  * If row == column, we add the jitter to diagonal

  * The next loop subtracts previously computed contributions

  * The if conditional checks positivity as cholesky requires positive diagonals

  * The else extracts the diagonal element as a normalization factor and the if prevents division by zero

  * If decomposition succeeded, it exits immediately, the jitter stabilizes the matrix by increasing jitter.

  * If all failed it returns false

* The next function is the `multivariate` which is the sampler:

  * Its arguments are:

  * rng: `std::mt19937&` seeded once outside and reused

  * mu0: `const std::array<double,dim>&`

  * Sigma: `const std::array<double,dim*dim>&` (flat row-major)

  * A standard normal distribution N(0,1) is used to generate random Gaussian values for sampling.

  * If covariance contained invalid values, sampling is impossible so its fallback is it returns mu0 values

  * We symmetrize the sigma because cholesky works with symmetry.

  * Floating point rounding can make matrix slightly asymmetric and we store this in a flat array called Sigma_sym.

  * Next, we have L where cholesky is computed with jitter attempts.

  * If bool chol_ok returns false, then it falls back to diagonal covariance (diagonal sqrt of diag(Sigma))

  * Then we generate samples and keep them in Z vector. The clamp is also used here to keep things in check.

  * Then the transformation is done to compute each coordinate, store `L*z`. The contribution from dimension j is multiplied by `z[j]` of that position.

  * There is a check that avoids propagation of invalid numbers.

  * The final is set, if not finite, then we use mean as a fallback.

  * Then X_flat is set using its navigation map, and `val` is officially it.

5. `step()`

* Now we go back to step specifically, we have officially generated our samples and stored them in a 1d array(`X_flat`)
* We also prevent numerical issues that may create inf values here too, so that it doesn't break clustering and weights later. The fallback is specifically mean for that coordinate.

  * Next, we specifically evaluate the objective function for each sample in the batch in this line:
    `std::array<double, batch> fvals = fobj<dim, batch>(X_flat)`

  * We store this in fvals of batch size, each batch outputted its own.

  * Our next step involves try_refocus which basically is responsible for maintaining an adaptive refocus on promising regions.

  * Its called in here: `_try_refocus(X_flat, fvals, refocus_thres, refocus_min_samples); `

  * Lets get into it:

6. `_try_refocus`

* Sometimes optimization converges prematurely, this function tries to detect possible optimums and refocus the search
* Returns a pair, first which is whether refocus happened and second which is the best objective value

  * We initialize count to 0.

  * We gather indices of near zero samples by creating a vector indices(`We cannot use arrays here before we are not certain of the indices size at compile time so the best we use vector`)

  * we also used indices.push_back to store which is more cleaner.

  * Basically if fvals[i] is more than or equal to refocus_threshold which the user set, we record the index.  The rationale behind this is based on the problem being solved here, it's a `10D Negated Rastrigin Problem`. So the initial iterations typically start at larger negative values, and on progression they become more and more positive. That is what we have to think about. Meaning the fval is less negative than our threshold which effectively means a set optimum has been reached in these occasions.

  * The count is `indices.size()`. We had initialized this earlier.

  * Now, here is where the magic happens:

  * `(count >= refocus_min_samples)` -> The goal here is that if the count is more than or equal to the refocus min samples we had initially come up with, then we ought to process new means and sigmas, focused on those samples.

  * We come up with a `std::vector<double> X0(count * dim, 0.0)` which has size count * dim and is initialized to 0

  * Then from X_flat which were our initial samples, we extract the indices using the X_flat mapping rule so that we copy those specific indices recorded earlier into X0.

  * We then officially compute a mean of promising samples and store them in mu0;

  * We also compute the covariance of promising samples and store them in sigma_new

  * Then we update the mu0 and sigma0 with mu0_new and sigma0_new

  * then we compute the best objective and then we return the bool true together with best_f.

  * If all of these had failed, specifically there were no indices that fulfilled the condition then it would have been terminated, and then for the second one, where the number of samples did not fulfill, then we would have effectively also returned a `{false, 0.0}`.

7. `step()`

* The step specifically checks to see if it returned a true, if it did not return a true, then no update will be done.
* Next we move on to macro cluster step:
* We create a vector<bool> of `mask_macro` which will show the specific indices needed for the macro.
* Lets get into it:

8. `std::vector<bool> cluster_mask_macro(const std::array<double, batch*dim> &X_flat, double tau)`

* This one has arguments X_flat which are the original samples as well as double tau which is the tau_macro threshold which determines the most dominant region threshold.

  * We start off by creating N which is the size of batch specifically.

  * then we create double sq the size of N*N which is essentially the size batch * batch and we initialize it with 0.0

  * We then compute the pairwise distances - we went with a vector as compared to an array due to segmentation faults (stack overflow)

  * This loop computes the Euclidean squared distances between all pairs of samples in X_flat.

  * Next we compute the median distance which we use as kernel width h, we copy sq into a newly created vector `sq_copy`, we create the mid point `auto mid_it` and then we find the median which we call `double median_sq` which is then pointed to by *mid_it
    then we calculate `kernel width h` by finding the square root of the median_sq then add `1e-12` which effectively prevents a division by zero

  * Next, we compute the `similarity matrix`. This stage converts distance to similarity matrix using a Gaussian kernel. close points are more and more closer to 1 and further points are more and more closer to 0.

  * Next, we find the median in the similarity matrix and it ends up being pointed by *mid2, then the official threshold is `double thresh = tau * median_S`

  * After that we find the neighbours of each point and record it. The neighbours connection must be equal to or more than the threshold `double_thresh`

  * Then in the final one we use Breadth-First Search to find the largest connected component in the list of neighbours. We use `deque` here.

  * Next, we build a boolean mask of size N which contains the indices for best comp which are specifically labeled true in the whole, which means we now have a mask where false is non clusters and true is for clusters.

  * That is returned.

9. `step()`

* This bool is then stored in `std::vector<bool> mask_macro`

* We then calculate how many mask macros are there in the bool

* We then create Xm and fm.

* If `macro_count == 0;` then we officially fallback to using the entire batch else if not, we use `macro_count * dim`.

* We extract those specific indices from `X_flat` to store in `Xm` and for the `fvals` to store in `fm`

* We then do the sample for the micro with `cluster_mask_micro`. The difference with cluster_mask_micro is that unlike `cluster_mask_macro`, this one does not take `X_flat`, it instead takes the macro cluster which is `Xm`

* If micro is empty we then fall back to macro cluster Xm which if empty, we officially fall back to X_flat and fvals.

* Next is weighting, now we have the micro count and `Xc` and `fc`, we now want to create weights that specifically lead to a new sigma and mean.

  * We create `std::vector<double> log_w(micro_count), w(micro_count)` and `double max_log_w = -std::numeric_limits<double>::infinity()`.

  * Here eta comes into play, it's a weight scaling factor.

  * We begin by checking if number is finite, if it is not, we use -1e300 to prevent an overflow.

  * If it is finite we calculate the log weights by `log_w[i] = eta * fc[i]`

  * If then we favour high objective values by setting this: `if (log_w[i] > max_log_w) max_log_w = log_w[i];`

  * Next is the creation of normalized probabilities:

    * Create `double sum_w` and initialize to 0.0.
    * We then exponentiate the difference between all log_w - max_log_w and accumulate the sum in `sum_w` and the individuals in `w`.
    * If the sum_w is not a positive definite value, we have a fallback to avoid numerical issues.
    * The fallback uses a uniform weight gotten by `1.0 / static_cast<double>(micro_count)`
    * If it does not fulfil, then we normalize the weights using `sum_w`
    * Next is the computation of the new mean. We create new_mu and new_sigma and initialize them.
    * For `new_mu`, we use w[i] to modify and bias it and the same goes for `new_sigma`
    * We still do add jitter for numerical safety, and also we make sure to validate the new_sigma to avoid instability issues

* Next is we commit the new updates to mu0 and  sigma0; and we return the best f.

* Remember the step is a pair with a bool and a double so we return:

```cpp
{false, best_f};
```

And that is a full optimization step.

10. `optimize(int iters)`

```cpp
std::vector<HistoryEntry> optimize(int iters){
std::vector<HistoryEntry> history;
history.reserve(iters);
    for(int i = 0;i<iters;i++){
        auto r = step();

        HistoryEntry h;
        h.mu=mu0;
        h.best_f = r.second;
        h.refocused=r.first;
        history.push_back(std::move(h));
    }
    return history;
```

* In the code above, we have already covered the `auto r = step()` which produces a pair of bool and double.
* `HistoryEntry h` collects the mu0, best_f, r.second and r.first and accumulates them. Once completed, it is all printed specifically in run:

```cpp
auto history = optimize(iterations);
for (size_t i = 0; i < history.size(); ++i) {
    const auto& h = history[i];
    std::cout << "Iter " << i<< " best_f=" << h.best_f << " mu=(";
    for (int d = 0; d < dim; ++d) {
        std::cout << h.mu[d] << (d+1<dim? ", ":"");
    }
    std::cout << ") refocused=" << (h.refocused? "yes":"no") << "\n";
}
```

## License

MIT License © 2026 Tom Muga
