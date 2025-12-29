"""
    LightweightStats

A lightweight Julia package providing basic statistical functions with zero dependencies.

This package offers drop-in replacements for the most commonly used functions from Statistics.jl,
but without any dependencies beyond Julia's standard library. It's ideal for projects that need
basic statistical operations while minimizing their dependency footprint.

# Exported Functions
- [`mean`](@ref) - Compute arithmetic mean
- [`median`](@ref) - Compute median value
- [`std`](@ref) - Compute standard deviation  
- [`var`](@ref) - Compute variance
- [`cov`](@ref) - Compute covariance
- [`cor`](@ref) - Compute correlation
- [`quantile`](@ref) - Compute quantiles
- [`middle`](@ref) - Compute middle of a range

# Examples
```julia
using LightweightStats

x = [1, 2, 3, 4, 5]
mean(x)      # 3.0
std(x)       # 1.5811388300841898
median(x)    # 3
```
"""
module LightweightStats

export mean, median, std, var, cov, cor, quantile, middle

# Internal implementation of complex conjugate to avoid LinearAlgebra dependency
_conj(x::Real) = x
_conj(x::Complex) = Complex(real(x), -imag(x))
_conj(x::AbstractArray{<:Real}) = x
_conj(x::AbstractArray{<:Complex}) = _conj.(x)

"""
    mean(A::AbstractArray; dims=:)

Compute the arithmetic mean of array `A`.

# Arguments
- `A::AbstractArray`: Input array
- `dims=:`: Dimensions along which to compute the mean. 
  - `:` (default) computes mean of entire array
  - Integer or tuple specifies dimensions

# Returns
- Scalar mean when `dims=:`
- Array of means along specified dimensions otherwise

# Examples
```jldoctest
julia> using LightweightStats

julia> mean([1, 2, 3, 4, 5])
3.0

julia> mean([1 2 3; 4 5 6])
3.5

julia> mean([1 2 3; 4 5 6]; dims=1)
1×3 Matrix{Float64}:
 2.5  3.5  4.5

julia> mean([1 2 3; 4 5 6]; dims=2)
2×1 Matrix{Float64}:
 2.0
 5.0
```

# Errors
Throws `ArgumentError` if array is empty.
"""
function mean(A::AbstractArray; dims = :)
    if dims === (:)
        isempty(A) && throw(ArgumentError("mean of empty collection undefined"))
        return sum(A) / length(A)
    else
        return mapslices(mean, A; dims = dims)
    end
end

"""
    mean(f, A::AbstractArray)

Apply function `f` to each element of `A` and compute the mean of the results.

# Arguments
- `f`: Function to apply to each element
- `A::AbstractArray`: Input array

# Returns
Mean of `f` applied to elements of `A`

# Examples
```jldoctest
julia> using LightweightStats

julia> mean(x -> x^2, [1, 2, 3])
4.666666666666667

julia> mean(abs, [-1, 0, 1])
0.6666666666666666
```

# Errors
Throws `ArgumentError` if array is empty.
"""
function mean(f, A::AbstractArray)
    isempty(A) && throw(ArgumentError("mean of empty collection undefined"))
    return sum(f, A) / length(A)
end

"""
    median(v::Vector)

Compute the median of vector `v`.

The median is the middle value when the data is sorted. For even-length vectors,
returns the average of the two middle values.

Note: This function is restricted to `Vector` type for performance reasons
as it uses scalar indexing.

# Arguments
- `v::Vector`: Input vector

# Returns
- The median value

# Examples
```jldoctest
julia> using LightweightStats

julia> median([1, 2, 3, 4, 5])
3

julia> median([1, 2, 3, 4])
2.5

julia> median([3, 1, 2])
2
```

# Errors
Throws `ArgumentError` if vector is empty.
"""
function median(v::Vector)
    isempty(v) && throw(ArgumentError("median of empty collection undefined"))
    sorted = sort(v)
    n = length(sorted)
    if isodd(n)
        return sorted[(n + 1) ÷ 2]
    else
        return (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
    end
end

"""
    median(A::Array; dims=:)

Compute the median of array `A`, optionally along specified dimensions.

Note: This function is restricted to `Array` type for performance reasons
as it uses scalar indexing.

# Arguments
- `A::Array`: Input array
- `dims=:`: Dimensions along which to compute median
  - `:` (default) computes median of entire array
  - Integer or tuple specifies dimensions

# Returns
- Scalar median when `dims=:`
- Array of medians along specified dimensions otherwise

# Examples
```jldoctest
julia> using LightweightStats

julia> median([1 2 3; 4 5 6])
3.5

julia> A = [1 2 3; 4 5 6]
2×3 Matrix{Int64}:
 1  2  3
 4  5  6

julia> median(A; dims=1)
1×3 Matrix{Float64}:
 2.5  3.5  4.5

julia> median(A; dims=2)
2×1 Matrix{Int64}:
 2
 5
```
"""
function median(A::Array; dims = :)
    if dims === (:)
        return median(vec(A))
    else
        return mapslices(median, A; dims = dims)
    end
end

"""
    var(A::AbstractArray; corrected=true, mean=nothing, dims=:)

Compute the variance of array `A`.

# Arguments
- `A::AbstractArray`: Input array
- `corrected::Bool=true`: If true, uses Bessel's correction (divides by n-1)
- `mean=nothing`: Pre-computed mean (for efficiency). If nothing, computes mean internally
- `dims=:`: Dimensions along which to compute variance

# Returns
- Scalar variance when `dims=:`
- Array of variances along specified dimensions otherwise
- Returns `NaN` for single-element arrays when `corrected=true`

# Examples
```jldoctest
julia> using LightweightStats

julia> var([1, 2, 3, 4, 5])
2.5

julia> var([1, 2, 3, 4, 5]; corrected=false)
2.0

julia> var([1, 2, 3, 4, 5]; mean=3)
2.5

julia> var([1 2 3; 4 5 6]; dims=1)
1×3 Matrix{Float64}:
 4.5  4.5  4.5
```

# Mathematical Definition
For corrected variance (default):
```math
s² = \\frac{1}{n-1} \\sum_{i=1}^{n} (x_i - \\bar{x})²
```

For uncorrected variance:
```math
σ² = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\bar{x})²
```
"""
function var(A::AbstractArray; corrected::Bool = true, mean = nothing, dims = :)
    if dims === (:)
        n = length(A)
        isempty(A) && return oftype(real(zero(eltype(A)))/1, NaN)
        m = mean === nothing ? LightweightStats.mean(A) : mean
        s = sum(x -> abs2(x - m), A)
        return corrected ? s / (n - 1) : s / n
    else
        return mapslices(x -> var(x; corrected = corrected, mean = mean), A; dims = dims)
    end
end

"""
    std(A::AbstractArray; corrected=true, mean=nothing, dims=:)

Compute the standard deviation of array `A`.

Standard deviation is the square root of variance.

# Arguments
- `A::AbstractArray`: Input array
- `corrected::Bool=true`: If true, uses Bessel's correction
- `mean=nothing`: Pre-computed mean (for efficiency)
- `dims=:`: Dimensions along which to compute standard deviation

# Returns
- Scalar standard deviation when `dims=:`
- Array of standard deviations along specified dimensions otherwise

# Examples
```jldoctest
julia> using LightweightStats

julia> std([1, 2, 3, 4, 5])
1.5811388300841898

julia> std([1, 2, 3, 4, 5]; corrected=false)
1.4142135623730951

julia> A = [1 2 3; 4 5 6];

julia> std(A)
1.8708286933869707

julia> std(A; dims=1)
1×3 Matrix{Float64}:
 2.12132  2.12132  2.12132
```
"""
function std(A::AbstractArray; corrected::Bool = true, mean = nothing, dims = :)
    return dims === (:) ? sqrt(var(A; corrected = corrected, mean = mean)) :
           sqrt.(var(A; corrected = corrected, mean = mean, dims = dims))
end

"""
    cov(x::AbstractVector, y::AbstractVector; corrected=true)

Compute the covariance between vectors `x` and `y`.

Supports complex-valued vectors using internal complex conjugate implementation.

# Arguments
- `x::AbstractVector`: First vector
- `y::AbstractVector`: Second vector  
- `corrected::Bool=true`: If true, uses Bessel's correction

# Returns
Scalar covariance value

# Examples
```jldoctest
julia> using LightweightStats

julia> x = [1.0, 2.0, 3.0, 4.0, 5.0];

julia> y = [2.0, 4.0, 6.0, 8.0, 10.0];

julia> cov(x, y)
5.0

julia> cov(x, y; corrected=false)
4.0
```

# Errors
Throws `DimensionMismatch` if vectors have different lengths.
"""
function cov(x::AbstractVector, y::AbstractVector; corrected::Bool = true)
    n = length(x)
    length(y) == n || throw(DimensionMismatch("x and y must have the same length"))
    n == 0 && return oftype(real(zero(eltype(x)))/1, NaN)

    xmean = mean(x)
    ymean = mean(y)

    # Use loop to avoid allocating intermediate arrays
    T = promote_type(typeof(xmean), typeof(ymean))
    s = zero(T)
    @inbounds @simd for i in eachindex(x, y)
        s += (x[i] - xmean) * _conj(y[i] - ymean)
    end
    return corrected ? s / (n - 1) : s / n
end

"""
    cov(x::AbstractVector; corrected=true)

Compute the variance of vector `x` (self-covariance).

# Arguments
- `x::AbstractVector`: Input vector
- `corrected::Bool=true`: If true, uses Bessel's correction

# Returns
Variance of the vector

# Examples
```jldoctest
julia> using LightweightStats

julia> cov([1, 2, 3, 4, 5])
2.5
```
"""
function cov(x::AbstractVector; corrected::Bool = true)
    return var(x; corrected = corrected)
end

"""
    cov(X::AbstractMatrix; dims=1, corrected=true)

Compute the covariance matrix of `X`.

Supports complex-valued matrices using internal complex conjugate implementation.
Uses broadcasting for better performance.

# Arguments
- `X::AbstractMatrix`: Data matrix
- `dims::Int=1`: Dimension along which variables are organized
  - `dims=1`: Each column is a variable (default)
  - `dims=2`: Each row is a variable
- `corrected::Bool=true`: If true, uses Bessel's correction

# Returns
Covariance matrix

# Examples
```jldoctest
julia> using LightweightStats

julia> X = [1 2; 3 4; 5 6]
3×2 Matrix{Int64}:
 1  2
 3  4
 5  6

julia> cov(X; dims=1)
2×2 Matrix{Float64}:
 4.0  4.0
 4.0  4.0

julia> cov(X; dims=2)
3×3 Matrix{Float64}:
 0.5  0.5  0.5
 0.5  0.5  0.5
 0.5  0.5  0.5
```

# Errors
Throws `ArgumentError` if dims is not 1 or 2.
"""
function cov(X::AbstractMatrix; dims::Int = 1, corrected::Bool = true)
    if dims == 1
        n, p = size(X)
        n == 0 && return fill(oftype(real(zero(eltype(X)))/1, NaN), p, p)

        T = float(real(eltype(X)))

        # Compute means with a loop to avoid allocations
        means = zeros(T, p)
        @inbounds for j in 1:p
            s = zero(eltype(X))
            @simd for i in 1:n
                s += X[i, j]
            end
            means[j] = real(s) / n
        end

        # Allocate output
        C = zeros(T, p, p)
        divisor = corrected ? n - 1 : n

        # Compute covariance with nested loops (upper triangle only)
        @inbounds for j in 1:p
            for k in j:p
                s = zero(T)
                @simd for i in 1:n
                    s += real((X[i, j] - means[j]) * _conj(X[i, k] - means[k]))
                end
                C[j, k] = s / divisor
                if j != k
                    C[k, j] = C[j, k]
                end
            end
        end

        return C
    elseif dims == 2
        n, p = size(X)
        p == 0 && return fill(oftype(real(zero(eltype(X)))/1, NaN), n, n)

        T = float(real(eltype(X)))

        # Compute means for rows
        means = zeros(T, n)
        @inbounds for i in 1:n
            s = zero(eltype(X))
            @simd for j in 1:p
                s += X[i, j]
            end
            means[i] = real(s) / p
        end

        # Allocate output
        C = zeros(T, n, n)
        divisor = corrected ? p - 1 : p

        # Compute covariance with nested loops (upper triangle only)
        @inbounds for j in 1:n
            for k in j:n
                s = zero(T)
                @simd for i in 1:p
                    s += real((X[j, i] - means[j]) * _conj(X[k, i] - means[k]))
                end
                C[j, k] = s / divisor
                if j != k
                    C[k, j] = C[j, k]
                end
            end
        end

        return C
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end
end

"""
    cor(x::AbstractVector, y::AbstractVector)

Compute the Pearson correlation coefficient between vectors `x` and `y`.

The correlation coefficient measures the linear relationship between two variables,
ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).

# Arguments
- `x::AbstractVector`: First vector
- `y::AbstractVector`: Second vector

# Returns
Correlation coefficient in range [-1, 1], or NaN if either vector has zero variance

# Examples
```jldoctest
julia> using LightweightStats

julia> x = [1.0, 2.0, 3.0, 4.0, 5.0];

julia> y = [2.0, 4.0, 6.0, 8.0, 10.0];

julia> cor(x, y)  # Perfect positive correlation
0.9999999999999998

julia> cor(x, -y)  # Perfect negative correlation
-0.9999999999999998

julia> cor(x, [1, 1, 1, 1, 1])  # No variance in y
NaN
```

# Mathematical Definition
```math
r = \\frac{\\text{cov}(x, y)}{\\sigma_x \\sigma_y}
```

# Errors
Throws `DimensionMismatch` if vectors have different lengths.
"""
function cor(x::AbstractVector, y::AbstractVector)
    n = length(x)
    length(y) == n || throw(DimensionMismatch("x and y must have the same length"))
    n == 0 && return oftype(real(zero(eltype(x)))/1, NaN)

    # Compute means
    xmean = mean(x)
    ymean = mean(y)

    # Compute covariance and variances in one pass to avoid redundant iterations
    cov_xy = zero(promote_type(typeof(xmean), typeof(ymean)))
    var_x = zero(typeof(xmean))
    var_y = zero(typeof(ymean))

    @inbounds @simd for i in eachindex(x, y)
        dx = x[i] - xmean
        dy = y[i] - ymean
        cov_xy += dx * _conj(dy)
        var_x += abs2(dx)
        var_y += abs2(dy)
    end

    (var_x == 0 || var_y == 0) && return oftype(real(zero(eltype(x)))/1, NaN)

    return cov_xy / sqrt(var_x * var_y)
end

"""
    cor(X::AbstractMatrix; dims=1)

Compute the correlation matrix of `X`.

Uses broadcasting for better performance.

# Arguments
- `X::AbstractMatrix`: Data matrix
- `dims::Int=1`: Dimension along which variables are organized
  - `dims=1`: Each column is a variable (default)
  - `dims=2`: Each row is a variable

# Returns
Correlation matrix where diagonal elements are 1 (or NaN for zero-variance variables)

# Examples
```jldoctest
julia> using LightweightStats

julia> X = [1 2; 3 4; 5 6]
3×2 Matrix{Int64}:
 1  2
 3  4
 5  6

julia> cor(X; dims=1)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0

julia> cor([1 2 3; 1 2 3]; dims=2)  # Identical rows
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```
"""
function cor(X::AbstractMatrix; dims::Int = 1)
    if dims == 1
        n, p = size(X)
        n == 0 && return fill(oftype(real(zero(eltype(X)))/1, NaN), p, p)

        T = float(real(eltype(X)))

        # Compute means with a loop
        means = zeros(T, p)
        @inbounds for j in 1:p
            s = zero(eltype(X))
            @simd for i in 1:n
                s += X[i, j]
            end
            means[j] = real(s) / n
        end

        # Compute standard deviations
        stds = zeros(T, p)
        @inbounds for j in 1:p
            s = zero(T)
            @simd for i in 1:n
                s += abs2(X[i, j] - means[j])
            end
            stds[j] = sqrt(s / n)
        end

        # Allocate output
        R = zeros(T, p, p)
        nan_val = oftype(T(0)/1, NaN)

        # Compute correlation with nested loops (upper triangle only)
        @inbounds for j in 1:p
            for k in j:p
                if stds[j] == 0 || stds[k] == 0
                    R[j, k] = nan_val
                    if j != k
                        R[k, j] = nan_val
                    end
                else
                    s = zero(T)
                    @simd for i in 1:n
                        s += real((X[i, j] - means[j]) * _conj(X[i, k] - means[k]))
                    end
                    R[j, k] = s / (n * stds[j] * stds[k])
                    if j != k
                        R[k, j] = R[j, k]
                    end
                end
            end
        end

        return R
    elseif dims == 2
        n, p = size(X)
        p == 0 && return fill(oftype(real(zero(eltype(X)))/1, NaN), n, n)

        T = float(real(eltype(X)))

        # Compute means for rows
        means = zeros(T, n)
        @inbounds for i in 1:n
            s = zero(eltype(X))
            @simd for j in 1:p
                s += X[i, j]
            end
            means[i] = real(s) / p
        end

        # Compute standard deviations for rows
        stds = zeros(T, n)
        @inbounds for i in 1:n
            s = zero(T)
            @simd for j in 1:p
                s += abs2(X[i, j] - means[i])
            end
            stds[i] = sqrt(s / p)
        end

        # Allocate output
        R = zeros(T, n, n)
        nan_val = oftype(T(0)/1, NaN)

        # Compute correlation with nested loops (upper triangle only)
        @inbounds for j in 1:n
            for k in j:n
                if stds[j] == 0 || stds[k] == 0
                    R[j, k] = nan_val
                    if j != k
                        R[k, j] = nan_val
                    end
                else
                    s = zero(T)
                    @simd for i in 1:p
                        s += real((X[j, i] - means[j]) * _conj(X[k, i] - means[k]))
                    end
                    R[j, k] = s / (p * stds[j] * stds[k])
                    if j != k
                        R[k, j] = R[j, k]
                    end
                end
            end
        end

        return R
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end
end

"""
    quantile(v::Vector, p::Real)

Compute the `p`-th quantile of vector `v`.

Uses linear interpolation between sorted values for non-exact quantile positions.

Note: This function is restricted to `Vector` type for performance reasons
as it uses scalar indexing.

# Arguments
- `v::Vector`: Input vector
- `p::Real`: Quantile value in range [0, 1]

# Returns
The p-th quantile value

# Examples
```jldoctest
julia> using LightweightStats

julia> v = [1, 2, 3, 4, 5];

julia> quantile(v, 0.0)  # Minimum
1

julia> quantile(v, 0.25)  # First quartile
2.0

julia> quantile(v, 0.5)  # Median
3.0

julia> quantile(v, 0.75)  # Third quartile
4.0

julia> quantile(v, 1.0)  # Maximum
5
```

# Errors
- Throws `ArgumentError` if `p` is not in [0, 1]
- Throws `ArgumentError` if vector is empty
"""
function quantile(v::Vector, p::Real)
    0 <= p <= 1 || throw(ArgumentError("quantile requires 0 <= p <= 1"))
    isempty(v) && throw(ArgumentError("quantile of empty collection undefined"))

    sorted = sort(v)
    n = length(sorted)
    T = float(eltype(v))

    if p == 0
        return sorted[1]
    elseif p == 1
        return sorted[n]
    else
        h = T((n - 1) * p + 1)
        i = floor(Int, h)
        if i == n
            return sorted[n]
        else
            return sorted[i] + (h - i) * (sorted[i + 1] - sorted[i])
        end
    end
end

"""
    quantile(v::Vector, p::AbstractVector)

Compute multiple quantiles of vector `v`.

Note: This function is restricted to `Vector` type for performance reasons
as it uses scalar indexing.

# Arguments
- `v::Vector`: Input vector
- `p::AbstractVector`: Vector of quantile values in range [0, 1]

# Returns
Vector of quantile values corresponding to each p

# Examples
```jldoctest
julia> using LightweightStats

julia> v = [1, 2, 3, 4, 5];

julia> quantile(v, [0.25, 0.5, 0.75])  # Quartiles
3-element Vector{Float64}:
 2.0
 3.0
 4.0

julia> quantile(v, 0:0.25:1)  # All quartiles including extremes
5-element Vector{Real}:
 1
 2.0
 3.0
 4.0
 5
```
"""
function quantile(v::Vector, p::AbstractVector)
    all(x -> 0 <= x <= 1, p) || throw(ArgumentError("quantile requires 0 <= p <= 1"))
    isempty(v) && throw(ArgumentError("quantile of empty collection undefined"))

    sorted = sort(v)
    n = length(sorted)
    T = float(eltype(v))

    result = Vector{T}(undef, length(p))

    @inbounds for (idx, pi) in enumerate(p)
        if pi == 0
            result[idx] = sorted[1]
        elseif pi == 1
            result[idx] = sorted[n]
        else
            h = T((n - 1) * pi + 1)
            i = floor(Int, h)
            if i == n
                result[idx] = sorted[n]
            else
                result[idx] = sorted[i] + (h - i) * (sorted[i + 1] - sorted[i])
            end
        end
    end

    return result
end

"""
    middle(x::Real, y::Real)

Compute the middle value between two numbers.

# Arguments
- `x::Real`: First number
- `y::Real`: Second number

# Returns
The average of x and y

# Examples
```jldoctest
julia> using LightweightStats

julia> middle(1, 5)
3.0

julia> middle(-10, 10)
0.0

julia> middle(2.5, 3.5)
3.0
```
"""
function middle(x::Real, y::Real)
    return (x + y) / 2
end

"""
    middle(a::AbstractArray)

Compute the middle of the range of values in array `a`.

Returns the midpoint between the minimum and maximum values.

# Arguments
- `a::AbstractArray`: Input array

# Returns
The middle value of the range

# Examples
```jldoctest
julia> using LightweightStats

julia> middle([1, 2, 3, 4, 5])
3.0

julia> middle([5, 1, 3])  # Order doesn't matter
3.0

julia> middle([-10, 0, 20])
5.0
```

# Errors
Throws `ArgumentError` if array is empty.
"""
function middle(a::AbstractArray)
    isempty(a) && throw(ArgumentError("middle of empty collection undefined"))
    return middle(extrema(a)...)
end

"""
    middle(x::Real)

Return the input value unchanged (identity function for single values).

# Arguments
- `x::Real`: Input value

# Returns
The same value

# Examples
```jldoctest
julia> using LightweightStats

julia> middle(42)
42

julia> middle(3.14)
3.14
```
"""
function middle(x::Real)
    return x
end

# Precompilation workload
using PrecompileTools

@compile_workload begin
    # Precompile the most commonly used functions with Float64 vectors
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]

    # Basic statistics on vectors
    mean(x)
    std(x)
    var(x)
    median(x)
    middle(x)
    quantile(x, 0.5)
    quantile(x, [0.25, 0.5, 0.75])

    # Two-vector operations
    cov(x, y)
    cor(x, y)

    # Matrix operations (common use case)
    X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    mean(X)
    mean(X; dims = 1)
    mean(X; dims = 2)
    std(X)
    var(X)
    median(X)
    cov(X)
    cov(X; dims = 2)
    cor(X)
end

end