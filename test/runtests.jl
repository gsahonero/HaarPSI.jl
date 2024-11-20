using HaarPSI
using Images
using Test

@testset "HaarPSI.jl" begin
    ref = Images.load(joinpath(pwd(), "images","r0.png"))
    dist = Images.load(joinpath(pwd(), "images","r1.png"))
    @test round(HaarPSI.HaarPSI_score(ref, ref))==1.0
end
