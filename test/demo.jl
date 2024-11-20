using Images
using HaarPSI

# Read images
ref = Images.load("./test/images/r0.png")
dist = Images.load("./test/images/r1.png")

# Calculate the perceptual quality score (VSI)
@time score = HaarPSI_score(ref, ref)

# Output the score
println("HaarPSI score: ", score)