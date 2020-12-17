# Neural Network Step by Step - Part II

by Timothy (tq.wang93@hotmail.com)

First draft: Dec, 2020

Last update: Dec, 2020



### Two-layer neural network

#### Neural network overview

#### Neural network representation

#### Feed-forward

* **For sample i **
$$
  \begin{align*}
  z^{[1]}_1 &= w^{[1]T}_1x + b^{[1]}_1\\
  a^{[1]}_1 &= \sigma(z^{[1]}_1)
  \end{align*}
  $$
  
  $$
  \begin{align*}
  z^{[1]}_2 &= w^{[1]T}_2x + b^{[1]}_2\\
  a^{[1]}_2 &= \sigma(z^{[1]}_2)
  \end{align*}
  $$
  
  
  For notation $a^{[l]}_i$, $l \gets layer$ and $i \gets node\,in\,layer$.
  $$
  \begin{align*}
  z^{[1]}_1 &= w^{[1]T}_1x + b^{[1]}_1, \quad a^{[1]}_1 = \sigma(z^{[1]}_1) \\[5pt]
  z^{[1]}_2 &= w^{[1]T}_2x + b^{[1]}_2, \quad a^{[1]}_2 = \sigma(z^{[1]}_2) \\[5pt]
  z^{[1]}_3 &= w^{[1]T}_3x + b^{[1]}_3, \quad a^{[1]}_3 = \sigma(z^{[1]}_3) \\[5pt]
  z^{[1]}_4 &= w^{[1]T}_4x + b^{[1]}_4, \quad a^{[1]}_4 = \sigma(z^{[1]}_4)
  \end{align*}
  $$
  
  $$
  \begin{align*}
  \newcommand{\cndash}{\rule{0.2em}{0pt}\rule[0.35em]{1.6em}{0.05em}\rule{0.2em}{0pt}}
  Z^{[1]} =&\begin{bmatrix}
               {\cndash} w^{[1]T}_1 {\cndash}\\[4pt]
               {\cndash} w^{[1]T}_2 {\cndash}\\[4pt]
               {\cndash} w^{[1]T}_3 {\cndash}\\[4pt]
               {\cndash} w^{[1]T}_4 {\cndash}
           \end{bmatrix}
           \cdot
           \begin{bmatrix}
               x1 \\
               x2 \\
               x3
           \end{bmatrix}
           +
           \begin{bmatrix}
               b^{[1]}_1 \\
               b^{[1]}_2 \\
               b^{[1]}_3 \\
               b^{[1]}_4
           \end{bmatrix}
           =
           \begin{bmatrix}
               w^{[1]T}_1x + b^{[1]}_1 \\
               w^{[1]T}_2x + b^{[1]}_2 \\
               w^{[1]T}_3x + b^{[1]}_3 \\
               w^{[1]T}_4x + b^{[1]}_4
           \end{bmatrix}
           =
           \begin{bmatrix}
           	 z^{[1]}_1 \\
           	 z^{[1]}_2 \\
           	 z^{[1]}_3 \\
           	 z^{[1]}_4
           \end{bmatrix} \\
           
           &\begin{array}{cc}
               \underbrace{\rule{39mm}{0mm}}
               \hspace{5.2em}
               \underbrace{\rule{12mm}{0mm}}
           \end{array} \\
          
           &\hspace{3em} W^{[1]}
            \hspace{8.5em} b^{[1]} \\[15pt]
  
  a^{[1]} &=\begin{bmatrix}
           	 a^{[1]}_1 \\
           	 a^{[1]}_2 \\
           	 a^{[1]}_3 \\
           	 a^{[1]}_4
           \end{bmatrix}
           =
           \sigma(Z^{[1]})
           
           
           
  \end{align*}
  $$
   
  
          
  
  Let's first define the vector representation of sample i and weight.
  $$
  \begin{gather*}
  
  x^{(i)} = 
          \begin{pmatrix}
              x^i_1\\
              x^i_2\\
              \vdots\\
              x^i_n\\
          \end{pmatrix}_{\textcolor{red}{n \times 1}} 
  
  \qquad
  
  w = 
          \begin{pmatrix}
              w_1\\
              w_2\\
              \vdots\\
              w_n\\
          \end{pmatrix}_{\textcolor{red}{n \times 1}}
  
  \end{gather*}
  $$
  We can then calculate z for sample i and represent it in a more concise way.

$$
\begin{align*}
z^{(i)} &= x^i_1w_1 + x^i_1w_1 + \dots + x^i_nw_n + b_i \\
        &= (w_1,w_2,\dots,w_n)
        \begin{pmatrix}
            x^i_1\\
            x^i_2\\
            \vdots\\
            x^i_n\\
        \end{pmatrix} + b_i\\
        &= w^Tx^{(i)} + b_i
\end{align*}
$$

* **For all samples**

$$
\begin{align*}

Z       &= [z^{(1)},z^{(2)},\dots,z^{(m)}]_{\textcolor{red}{1 \times m}}\\
        &= [w^Tx^{(1)}+b_1,w^Tx^{(2)}+b_2,\dots,w^Tx^{(m)}+b_m]\\
        &= w^T[x^{(1)},x^{(2)},\dots,x^{(m)}]+[b_1,b_2,\dots,b_m]\\
        &= w^TX + b \\[10pt]

w^T     &= [w_1,w_2,\dots,w_n]_{\textcolor{red}{1 \times n}}

\qquad

X        = 
         \begin{pmatrix}
             x^1_1 & x^2_1 & \dots & x^m_1\\
             x^1_2 & x^2_2 & \dots & x^m_2\\
             \vdots& \vdots& \vdots& \vdots\\
             x^1_n & x^2_n & \dots & x^m_n\\
         \end{pmatrix}_{\textcolor{red}{n \times m}}\\[10pt]

\hat{Y} &= sigmoid(Z) =
[\hat{y}^{(1)},\hat{y}^{(2)},\dots,\hat{y}^{(m)}]_{\textcolor{red}{1 \times m}}

\end{align*}
$$

#### Back propagation

Back propagation starts by comparing the **_predicted value $ \hat{y} $_** to the **_true value $ y $_**, which produces the **loss**. It then updates the weights $ w $ and bias $ b $ using **_gradient descent_**. Through iterations, the weights and bias are optimized. 

* **Loss function $L(a,y)$**

  Here, we use the **_cross entropy_** loss function. There are other options such as **_mean squared error_**.
  $$
  L(a,y) = -ylog\,a-(1-y)log(1-a)
  $$

* **Cost function $J(a,y)$**

  As defined, cost function is the **_mean_** of the loss of all samples.

$$
J(a,y) = J(w,b)=\frac{1}{m} \sum_{i=1}^{m}L(a,y)
$$

* **Gradient descent**

  *Based on chain rules,*
  $$
  \begin{align*}
  
        \frac{\partial L(a,y)}{\partial w} 
     &= \frac{\partial L(a,y)}{\partial a} \cdot
        \frac{\partial a}     {\partial z} \cdot
        \frac{\partial z}     {\partial w}
  
  \\[10pt]
  
        \frac{\partial L(a,y)}{\partial b} 
     &= \frac{\partial L(a,y)}{\partial a} \cdot
        \frac{\partial a}     {\partial z} \cdot
        \frac{\partial z}     {\partial b}
  
  \end{align*}
  $$
  **Note:** the *sigmoid* function is
  $$
  a = \sigma(z) = \frac{1}{1+e^{-z}}
  $$
  
  $$
  \begin{align*}
  
        \frac{\partial L(a,y)}{\partial a} 
      &= -y \, \frac{1}{a} - (1-y) \, \frac{1}{1-a}(-1)\\
      &= -\frac{y}{a} + \frac{1-y}{1-a}
  
  \\[15pt]
  
        \frac{\partial a}{\partial z} 
      &= \frac{0-1 \cdot e^{-z} \cdot (-1)}{(1 + e^{-z})^2}\\
      &= \frac{e^{-z}}{(1 + e^{-z})^2}\\
      &= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}\\
      &= \frac{1}{1 + e^{-z}} \cdot (1-\frac{1}{1 + e^{-z}})\\
      &= a \cdot (1-a)
  
  \end{align*}
  $$
  Therefore,
  $$
  \begin{align*}
    
     \frac{\partial L(a,y)}{\partial a} \cdot
     \frac{\partial a}     {\partial z}
     &= (-\frac{y}{a} + \frac{1-y}{1-a}) \, a(1-a)\\
     &= -y(1-a)+a(1-y)\\
     &= a-y
  
  \end{align*}
  $$
  

  Moving on, for the partial derivatives of $ z = w^T + b $  to $ w $ and $ b $,

  * In the case of **sample i**, because
    $$
    \begin{align*}
    
          \frac{\partial z}{\partial w} = x_{\textcolor{red}{n \times 1}}
          
          \qquad
          
          \frac{\partial x}{\partial b} = 1_{\textcolor{red}{1 \times 1}}
    
    
    
    
    \end{align*}
    $$
    we can get
    $$
    \begin{align*}
    
           \frac{\partial L(a,y)}{\partial w} 
        &= x_{\textcolor{red}{n \times 1}} (a-y)_{\textcolor{red}{1 \times 1}}
    
    \\[15pt]
    
           \frac{\partial L(a,y)}{\partial b} 
        &= (a-y)_{\textcolor{red}{1 \times 1}} \cdot 1_{\textcolor{red}{1 \times 1}}
    
    
    \end{align*}
    $$

  * In the case of **m samples**,
    $$
    \begin{align*}
    
           \frac{\partial L(A,Y)}{\partial w_{\textcolor{red}{n \times 1}}} 
        &= X_{\textcolor{red}{n \times m}} (A-Y)^T \, _{\textcolor{red}{m \times 1}}
    
    \\[15pt]
    
           \frac{\partial L(A,Y)}{\partial b_{\textcolor{red}{1 \times 1}}}
        &= A-Y \, _{\textcolor{red}{1 \times m}}\\
        &= np.sum(A-Y) \, _{\textcolor{red}{1 \times 1}}
    
    
    \end{align*}
    $$

* **Updating weights and bias**

  The weight $ w $ and bias $ b $ are updated by subtracting the **_mean of cost_** multiplied by the **_learning rate $ \alpha $_**.
  $$
  \begin{alignat}{0}
  
         w \, \gets \, w - \alpha \cdot \frac{1}{m} \cdot X(A-Y)^T  
  
  \\[10pt]
  
         b \, \gets \, b - \alpha \cdot \frac{1}{m} \cdot np.sum(A-Y)
  
  
  \end{alignat}
  $$

* **Revisiting the cost function J**

  Usually, we print/plot the change of cost as a function of iterations. 
  $$
  \begin{align*}
       
       J &= \frac{1}{m} \sum_{i=1}^{m} [-ylog\,a-(1-y)log(1-a)]\\
         &= -\frac{1}{m} \sum_{i=1}^{m} [ylog\,a+(1-y)log(1-a)]\\
         &= -\frac{1}{m} [Ylog\,A+(1-Y)log(1-A)]\\
         &= -\frac{1}{m} np.sum[Ylog\,A+(1-Y)log(1-A)]
  
  \end{align*}
  $$
  **Note:** the $ Ylog\,A+(1-Y)log(1-A) $ is calculated using dot product. 
