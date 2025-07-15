## Linear Regression

### Gradient Descent

- First we'll find the error.
$$ Error(E) = \sum_{k=1}^{N}(y^k - (w_0.x_0^k + w_1.x_1^k + ... + w_M.x_m^k))^2 $$
$$ = \sum_{k=1}^{N}(y^k - w.x^k)^2 = \sum_{k=1}^{N}\delta_k^2 $$
$$ where, \delta_k = y^k - w.x^k $$

The minimum of $E$ is reached when derivatives of $E$ w.r.t. each of the parameters $w_i$ is zero.

So, let's get the partial derivative $\rightarrow$

$$ \frac{\partial{E}}{\partial{w_i}} = -2\sum_{k=1}^{N}\delta_k.x_i^k $$

- Update Rule $\rightarrow$
We have to move in the direction opposite to that of the gradient
$$ w_i \leftarrow w_i -\alpha\frac{\partial{E}}{\partial{w_i}} $$
$$ \implies w_i \leftarrow w_i + \alpha\delta_kx_i^k$$  
where, $\alpha$ is the learning rate. 

#### <u>Stochastic Gradient Descent Algorithm</u>
- initialize $w_0, w_1, ..., w_m$
- set learning rate $\alpha$
- repeat for a number of epochs $(Ep)$:
- for each training example $(x^k, y^k)$:
    $$\overline{y} = \sum_{i = 0}^m (w_i * x_i^k)$$
    $$\delta_k = y^k - \overline{y}$$
- for i from 0 to m:
   $$ w_i = w_i + \alpha * \delta_k * x_i^k  $$
- Time Complexity $\rightarrow O(Ep.N.m)$