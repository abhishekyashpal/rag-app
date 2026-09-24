Math
 ↓
Neuron
 ↓
Linear regression
 ↓
Gradient descent
 ↓
Backpropagation
 ↓
MLP / FFN
 ↓
Embeddings
 ↓
Positional encoding
 ↓
Self-attention
 ↓
Causal attention
 ↓
Multi-head attention
 ↓
Residual connections
 ↓
LayerNorm
 ↓
Transformer block
 ↓
GPT architecture
 ↓
Next-token objective
 ↓
Cross-entropy
 ↓
Backprop through GPT
 ↓
Numerical gradient checking
 ↓
Adam
 ↓
Autoregressive generation
 ↓
Temperature / top-k













Mathematics you've actually used
Basic algebra
Variables
Expressions
Equations
Rearranging equations
Functions
Linear algebra
Scalars
Vectors
Matrices
Matrix multiplication
Transpose
Dot product
Matrix dimensions/shapes
Matrix-vector multiplication
Vector/matrix addition
Element-wise multiplication
Functions
Linear functions
Nonlinear functions
Composition of functions
Coordinate/geometric interpretation
Decision boundaries
Linear separability
Why a single linear neuron cannot solve XOR
Exponents
\(e^x\)
\(e^{-x}\)
Exponential scaling
Logarithms
Natural logarithm \(\ln(x)\)
\(\log\) probabilities
Log-sum-exp
Derivatives
Derivative of a function
Partial derivatives
Derivatives with respect to weights and biases
Derivative of loss
Multivariable calculus
Partial derivatives
Gradients
Gradient vectors
Gradients of matrices/tensors
Chain rule
Composition of derivatives
Computational graphs
Backpropagation
Optimization
Loss functions
Gradient descent
Learning rate
Parameter updates
Local optimization intuition
Probability
Probability distributions
Conditional probability
Next-token probability
Probability normalization

Softmax mathematics

$$ softmax(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}} $$

Cross-entropy

$$ L=-\log P_{\text{correct}} $$

Mean squared error

$$ L=(\hat y-y)^2 $$

and

$$ L=\frac12(\hat y-y)^2 $$
Statistics
Mean
Variance
Standard deviation
Normalization

Layer normalization mathematics

$$ \mu=\frac1d\sum_i x_i $$ $$ \sigma^2=\frac1d\sum_i(x_i-\mu)^2 $$ $$ \hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} $$
Similarity / inner products
Dot-product similarity
\(QK^T\) in attention
Scaling

Attention scaling:

$$ \frac{QK^T}{\sqrt{d_k}} $$
Numerical differentiation
Finite differences

Numerical gradient checking:

$$ \frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon} $$

Error/relative error analysis

$$ \frac{|a-n|} {\max(|a|,|n|,\epsilon)} $$
Optimization with momentum / moving averages
First moment
Second moment
Bias correction
Adam

Sequence probability / probability factorization

$$ P(x_1,\ldots,x_T) = \prod_t P(x_t|x_{<t}) $$

Temperature transformation of probabilities

$$ P_i= softmax\left(\frac{z_i}{T}\right) $$
Basic combinatorial/selection reasoning in sampling
Top-k selection
Renormalization of probabilities
Tensor/shape mathematics
\(B\times T\times d\)
Head dimensions
\(Q,K,V\) dimensions
Attention score dimensions
Broadcasting concepts



Pending----------

Training at Scale:
single example
    ↓
mini-batches
    ↓
GPU tensors
    ↓
mixed precision
    ↓
efficient attention
    ↓
gradient accumulation
    ↓
learning-rate schedules
    ↓
checkpointing
    ↓
distributed training
    ↓
multi-GPU

How much data?
How many parameters?
How much VRAM?
How many FLOPs?
How long will training take?

Inference at Scale:
trained model
    ↓
efficient inference
    ↓
KV cache
    ↓
batching
    ↓
quantization
    ↓
memory optimization
    ↓
throughput / latency
    ↓
serving
    ↓
API