# Testing nc_pca package on a simulated data
# trying mono and bi exp. decays

using Test
using NCPCA
using Noise
using TestImages
using Plots


@testset "NCPCA" begin

    # Importing a test image
    #u = testimage("bark_he_512")
    #u = Float32.(u)

    # OR have random data
    u = rand(Float32, (96, 96))
    
    
    # First, try with a mono-exp decay test
    # create exponential function
    t = collect( range(1, stop=70, step=1))
    exp_f_mono = exp.(-t/20) 
    exp_f_mono = Float32.(reshape(exp_f_mono, (1, 1, 70)))
    # added to the image
    test_u_mono = u .* exp_f_mono
    Data_mono = poisson(test_u_mono)
    #phasor plot 
    myhist_mono1, realPart_mono1, imagPart_mono1, mask_phasor_mono1 = phasorplot(Data_mono, 10, 0.2, true, 0)
    #Apply PCA
    mybasis_mono, eigenvectors_mono, scores_mono, mask_ncpca_mono, filt_img_mono, noise_correction_mono = nc_pca(Data_mono, 4, norm_poisson, -1.0, true)
    #Plot eigenvectors (Loadings)
    plot(collect(StepRange(1, Int8(1), 70)), eigenvectors_mono[:, 1], label = "PC 1")
    plot!(collect(StepRange(1, Int8(1), 70)), eigenvectors_mono[:, 2], label = "PC 2")
    plot!(collect(StepRange(1, Int8(1), 70)), eigenvectors_mono[:, 3], label = "PC 3")
    plot!(collect(StepRange(1, Int8(1), 70)), eigenvectors_mono[:, 4], label = "PC 4")
    # corrected phasor plot
    myhist_mono2, realPart_mono2, imagPart_mono2, mask_phasor_mono2 = phasorplot(filt_img_mono, 50, 0.2, true, 0)



    # Now, applied to bi-exp decay
    exp_f_bi = 0.7exp.(-t/20) + 0.3exp.(-t/50)
    exp_f_bi = Float32.(reshape(exp_f_bi, (1, 1, 70)))
    # added to the image
    test_u_bi = u .* exp_f_bi
    Data_bi = poisson(test_u_bi)
    #phasor plot 
    myhist_bi1, realPart_bi1, imagPart_bi1, mask_phasor_bi1 = phasorplot(Data_bi, 50, 0.2, true, 0)
    #Apply PCA
    mybasis_bi, eigenvectors_bi, scores_bi, mask_ncpca_bi, filt_img_bi, noise_correction_bi = nc_pca(Data_bi, 4, norm_poisson, -1.0, true)
    #Plot eigenvectors (Loadings)
    plot(collect(StepRange(1, Int8(1), 70)), eigenvectors_bi[:, 1], label = "PC 1")
    plot!(collect(StepRange(1, Int8(1), 70)), eigenvectors_bi[:, 2], label = "PC 2")
    plot!(collect(StepRange(1, Int8(1), 70)), eigenvectors_bi[:, 3], label = "PC 3")
    plot!(collect(StepRange(1, Int8(1), 70)), eigenvectors_bi[:, 4], label = "PC 4")
    # corrected phasor plot
    myhist_bi2, realPart_bi2, imagPart_bi2, mask_phasor_bi2 = phasorplot(filt_img_bi, 50, 0.2, true, 0)

    
    end



