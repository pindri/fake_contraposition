compute_y <- function(x, epsilon) {
  # Compute the CDF of the standard normal for x
  phi_x <- pnorm(x)
  # Calculate the corresponding y using the inverse CDF (qnorm)
  y <- qnorm(1 - epsilon / (1 - phi_x))
  return(y)
}
compute_y_uniform <- function(x, epsilon) {
  # Ensure x is within [0, 1) to avoid division by zero
  if (x >= 1) stop("x must be less than 1")

  # Compute the corresponding y
  y <- 1 - epsilon / (1 - x)

  return(y)
}
ineq <- function(n,eps, delta) {
    return(log((n * (n-1) + 2*n + 2)/2) + n * log(1 - eps) > log(delta))
}

ineq2_new <- function(n,eps,delta) {
    return(!(n >= 2/log(2)/(eps)*(log(1/delta) + 2*log(2*n)-log(1-exp(-n*eps/8)))))
}

ineq2 <- function(n,eps,delta) {
    return((n < 2/log(2)/eps*(log(2/delta) + 2*log(2*n))))
}

ineq3 <- function(n,eps,delta){
    return(n < (2/eps*(log(1/eps)+2*log(log(1/eps)) +3)))
}

complexity_orig <- function(eps,delta){
    return(16/eps*log(16/eps) + 2/eps*(2/delta))
}

ineq_fuck_it <- function(n, eps, delta, d = 2) {
    k <- n * eps / 2
    2^-k*(2*n)^d > delta * (1 - exp(-k/4))
#     exp(lfactorial(2*n - k) + lfactorial(n) - lfactorial(2*n) - lfactorial(n - k))
}



complexity <- function(eps,delta) {
    n <- 0
    increment <- 10^9
    while(increment >= 1){
        n <- n + ifelse(ineq2(n + increment, eps, delta),increment,0)
        print(increment)
        increment <- as.integer(increment / 2)
    }
    return(n)
}
# Set the value of epsilon
epsilon <- 0.0001  # Change this value if needed

# Generate a sequence of x values (e.g., from -3 to 3)
x_values <- seq(0, 0.9999, by = 0.00001)

# Compute the corresponding y values for each x
y_values <- sapply(x_values, compute_y_uniform, epsilon = epsilon)

# Create a data frame for plotting
data <- data.frame(x = x_values, y = y_values)
n <- 1*complexity(epsilon,0.5)
library(ggplot2)
library(dplyr)
bre <- runif(n)
sigmoid <- function(x){1/(1+exp(-x))}
sample <- data.frame(x= bre , y = runif(n))
sample <- sample %>% filter((x > 0.99) | (y > 0.99))
# sample <- sample
ggplot(data, aes(x = x, y = y)) +
  geom_line(color = "blue", size = 1) +
  labs(title = paste("Curve for P(X > x, Y > y) =", epsilon),
       x = "x",
       y = "y") +
  theme_minimal() + geom_point(data = sample, aes(x=x,y=y))