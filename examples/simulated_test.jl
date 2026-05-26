using TestImages
using Noise
# using FileIO
using NCPCA
using View5D

function main()
    obj = 1 .* Float32.(testimage("resolution_test_512"))
    obj_2 = obj[end:-1:1, :]

    N = 50
    tau1 = 12.0
    tau2 = 22.0
    spec_1 = reshape(exp.(- (1:N)./tau1), (1,1,N))
    spec_2 = reshape(exp.(- (1:N)./tau2), (1,1,N))

    data = obj .* spec_1 .+ obj_2 .* spec_2
    data_noise = poisson(data)
    @vv data_noise

    myhist, realPart, imagPart, mask_phasor = phasorplot(data_noise, 50, 0.2, true, 0)
    savefig("no_filtering.png")

    mybasis, eigenvectors, scores, mask_ncpca, filt_img, noise_correction = nc_pca(data_noise, 3, norm_poisson, -1.0, false)

    myhist2, realPart2, imagPart2, mask_phasor2 = phasorplot(filt_img, 50, 0.2, true, 0)
    savefig("filtering_3dims.png")

    myhist, realPart, imagPart, mask_phasor = phasorplot(data, 50, 0.2, true, 0)
    savefig("perfect.png")

    @vt data_noise filt_img
end
