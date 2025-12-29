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

    # Use broadcasting instead of scalar indexing
    s = sum((x .- xmean) .* _conj(y .- ymean))
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

        means = vec(mean(X; dims = 1))
        C = zeros(float(real(eltype(X))), p, p)
        
        # Center the data once using broadcasting
        X_centered = X .- means'

        for i in 1:p
            for j in i:p
                # Use views and broadcasting for column operations
                s = sum(view(X_centered, :, i) .* _conj(view(X_centered, :, j)))
                C[i, j] = corrected ? s / (n - 1) : s / n
                if i != j
                    C[j, i] = _conj(C[i, j])
                end
            end
        end
        return C
    elseif dims == 2
        n, p = size(X')
        n == 0 && return fill(oftype(real(zero(eltype(X)))/1, NaN), p, p)

        means = vec(mean(X; dims = 2))
        C = zeros(float(real(eltype(X))), p, p)
        
        # Center the data once using broadcasting
        X_centered = X .- means

        for i in 1:p
            for j in i:p
                # Use views and broadcasting for row operations
                s = sum(view(X_centered, i, :) .* _conj(view(X_centered, j, :)))
                C[i, j] = corrected ? s / (n - 1) : s / n
                if i != j
                    C[j, i] = _conj(C[i, j])
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
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))

    sx = std(x; corrected = false)
    sy = std(y; corrected = false)

    (sx == 0 || sy == 0) && return oftype(real(zero(eltype(x)))/1, NaN)

    return cov(x, y; corrected = false) / (sx * sy)
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
    C = cov(X; dims = dims, corrected = false)

    if dims == 1
        s = vec(std(X; dims = 1, corrected = false))
    else
        s = vec(std(X; dims = 2, corrected = false))
    end

    # Use broadcasting to compute correlation matrix
    # Create outer product of standard deviations
    s_outer = s * s'
    
    # Handle zero variance cases with broadcasting
    R = similar(C)
    zero_mask = (s_outer .== 0)
    R[zero_mask] .= oftype(real(zero(eltype(X)))/1, NaN)
    R[.!zero_mask] = C[.!zero_mask] ./ s_outer[.!zero_mask]

    return R
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
    return [quantile(v, pi) for pi in p]
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