
using Images
using DSP

function rgb2matrix(image)
    ref_img = permutedims(channelview(image),[2,3,1])[:, :, 1:3]
    ref_img = Float32.(ref_img)
    return ref_img
end

function HaarPSI_score(imgRef, imgDist; preprocessWithSubsampling=true)
    # HaarPSI Computes the Haar wavelet-based perceptual similarity index of two images.
    imgRef = rgb2matrix(imgRef)
    imgDist = rgb2matrix(imgDist)
    imgRef *= 256;
    imgDist *= 256;
    # Ensure the images are in the range [0, 255]
    imgRef = floor.(Float64.(imgRef))
    imgDist = floor.(Float64.(imgDist))

    # Check if the image is color (3 channels)
    colorImage = size(imgRef, 3) == 3
    
    # Initialize constants
    C = 30
    alpha = 4.2

    # Transform to YIQ color space for color images
    if colorImage
        imgRefY = 0.299 * imgRef[:, :, 1] + 0.587 * imgRef[:, :, 2] + 0.114 * imgRef[:, :, 3]
        imgDistY = 0.299 * imgDist[:, :, 1] + 0.587 * imgDist[:, :, 2] + 0.114 * imgDist[:, :, 3]
        imgRefI = 0.596 * imgRef[:, :, 1] - 0.274 * imgRef[:, :, 2] - 0.322 * imgRef[:, :, 3]
        imgDistI = 0.596 * imgDist[:, :, 1] - 0.274 * imgDist[:, :, 2] - 0.322 * imgDist[:, :, 3]
        imgRefQ = 0.211 * imgRef[:, :, 1] - 0.523 * imgRef[:, :, 2] + 0.312 * imgRef[:, :, 3]
        imgDistQ = 0.211 * imgDist[:, :, 1] - 0.523 * imgDist[:, :, 2] + 0.312 * imgDist[:, :, 3]
    else
        imgRefY = imgRef
        imgDistY = imgDist
    end
    
    # Preprocessing with subsampling
    if preprocessWithSubsampling
        imgRefY = HaarPSISubsample(imgRefY)
        imgDistY = HaarPSISubsample(imgDistY)
        if colorImage
            imgRefQ = HaarPSISubsample(imgRefQ)
            imgDistQ = HaarPSISubsample(imgDistQ)
            imgRefI = HaarPSISubsample(imgRefI)
            imgDistI = HaarPSISubsample(imgDistI)
        end
    end
    
    # Pre-allocate variables
    if colorImage
        localSimilarities = zeros(size(imgRefY)..., 3)
        weights = zeros(size(imgRefY)..., 3)
    else
        localSimilarities = zeros(size(imgRefY)..., 2)
        weights = zeros(size(imgRefY)..., 2)
    end
    
    # Haar wavelet decomposition
    nScales = 3
    coeffsRefY = HaarPSIDec(imgRefY, nScales)
    coeffsDistY = HaarPSIDec(imgDistY, nScales)
    
    if colorImage
        coeffsRefQ = abs.(conv(imgRefQ, ones(2, 2) / 4)[2:end, 2:end])
        coeffsDistQ = abs.(conv(imgDistQ, ones(2, 2) / 4)[2:end, 2:end])
        coeffsRefI = abs.(conv(imgRefI, ones(2, 2) / 4)[2:end, 2:end])
        coeffsDistI = abs.(conv(imgDistI, ones(2, 2) / 4)[2:end, 2:end])
    end
    
    # Compute weights and similarity for each orientation
    for ori in 1:2
        weights[:, :, ori] .= max.(abs.(coeffsRefY[:, :, 3 + (ori - 1) * nScales]), abs.(coeffsDistY[:, :, 3 + (ori - 1) * nScales]))
        coeffsRefYMag = abs.(coeffsRefY[:, :, (1:2) .+ (ori-1)*nScales])
        coeffsDistYMag = abs.(coeffsDistY[:, :, (1:2) .+ (ori-1)*nScales])
        localSimilarities[:, :, ori] .= sum((2 * coeffsRefYMag .* coeffsDistYMag .+ C) ./ (coeffsRefYMag .^ 2 .+ coeffsDistYMag .^ 2 .+ C), dims=3) / 2
    end
    
    # Compute similarities for color channels
    if colorImage
        similarityI = (2 * coeffsRefI .* coeffsDistI .+ C) ./ (coeffsRefI .^ 2 .+ coeffsDistI .^ 2 .+ C)
        similarityQ = (2 * coeffsRefQ .* coeffsDistQ .+ C) ./ (coeffsRefQ .^ 2 .+ coeffsDistQ .^ 2 .+ C)
        localSimilarities[:, :, 3] .= (similarityI .+ similarityQ) / 2
        weights[:, :, 3] .= (weights[:, :, 1] .+ weights[:, :, 2]) / 2
    end
    
    # Compute final score
    similarity = HaarPSILogInv(sum(HaarPSILog(localSimilarities[:] , alpha) .* weights[:])./ sum(weights[:]), alpha)^2
    
    # Output maps if requested
    similarityMaps = localSimilarities
    weightMaps = weights
    
    return similarity#, similarityMaps, weightMaps
end

# Haar wavelet decomposition
function HaarPSIDec(X, nScales)
    coeffs = zeros(size(X)..., 2 * nScales)
    for k in 1:nScales
        haarFilter = 2.0^(-k) * ones(2^k, 2^k)
        haarFilter[1:(end ÷ 2), :] .= -haarFilter[1:(end ÷ 2), :]
        kernel_size = size(haarFilter)
        coeffs[:, :, k] .= conv(X, haarFilter)[1+Int(kernel_size[1]/2):end-Int(kernel_size[1]/2-1), 1+Int(kernel_size[1]/2):end-Int(kernel_size[1]/2-1)]
        coeffs[:, :, k + nScales] .= conv(X, haarFilter')[1+Int(kernel_size[1]/2):end-Int(kernel_size[1]/2-1), 1+Int(kernel_size[1]/2):end-Int(kernel_size[1]/2-1)]
    end
    return coeffs
end

# Subsample the image
function HaarPSISubsample(img)
    imgSubsampled = conv(img, ones(2, 2) / 4)[2:end, 2:end]
    return imgSubsampled[1:2:end, 1:2:end]
end

# Logarithmic function for similarity computation
function HaarPSILog(x, alpha)
    return 1 ./ (1 .+ exp.(-alpha .* x))
end

# Inverse logarithmic function for similarity computation
function HaarPSILogInv(x, alpha)
    return log.(x ./ (1 .- x)) ./ alpha
end

export HaarPSI