using Statistics, LinearAlgebra, View5D, Plots

norm_poisson(mydata, eps=1e-10) = sqrt.(mean(mydata,dims=(1, 2)).+eps)
norm_variance(mydata, eps=1e-10) = std.(mydata,dims=(1,2)).+eps

"""
 nc_pca(mydata, max_dims=4, my_norm=norm_poisson, threshold=0, do_display=false)
 
 Performs NC-PCA on a series of images
 Authors: Rainer Heintzmann (heintzmann@gmail.com), Alix Le Marois (alix_le_marois@live.fr)
 INPUT ARGUMENTS
 mydata : Data to find the components from. 
           The "spectral" or "time" dimension along which the NCPCA is performed and Eigenvectors are found has to be oriented along the 3rd dimension. 
          Default value: HeLa cell image, figure 4 from the publication metioned below
 MaxDims : maximum dimensions to return (default value is 4)
 NormType :    - '' : no norm (equal variances are assumed), 
               - 'Poisson': NCPCA (default)
               - 'Variance' : normalize by standard deviations
 threshold : This can either be a binary mask of type 'dip_image' or a single value
             In the latter case, a mask will be calculated by selecting pixels above threshold for the maximum signal in dimension 3 of the dataset
             Default is a fixed threshold value of zero, i.e. selecting only positive pixels.
 doDiplay :  Boolean, if true, the Eigenvectors and the Scores will be displayed.
             Default: 0
 
 OUTPUT ARGUMENTS
 mybasis : principal components of scaled data
 Eigenvectors : rescaled principal components
 Scores : the corresponding abundance maps.
 mask_ncpca : if a threshold was applied; returns 1 in selected pixels and 0 elsewhere
 FiltImg : mydata projected onto the new subspace, so this is a denoised dataset
 NoiseCorrection : noise correction factors which were applied before performing a standard PCA

 NOTE: User must have the DIP_image image processing toolbox installed. It can be obtained at 
 http://www.diplib.org
 Our software is made available under the Gnu Public Licence Version II (GPL2)
 No warranty of any kind is given and users use it at their own risk.

 Please refer to and cite where appropriate: 
 Alix Le Marois, Simon Labouesse, Klaus Suhling, and Rainer Heintzmann, "Noise-corrected principal component analysis of fluorescence lifetime imaging (FLIM) data",
 Journal of Biophotonics (2016)

 Example 1 simple simulation
 a = 4*readim() .* exp(-zz([1 1 20])/3);    % simulate two images with different decays
 b = 2*readim('orka') .* exp(-zz([1 1 20])/3.1);
 c = noise(a + b,'Poisson');   % simulate Poisson statistics
 [myBasisS, EVS, ScoresS,maskS,FiltImgS]=NCPCA(c,4,'',0);  % Standard PCA
 ScoresS(:,:,2)  % Note that this component should contain no information, but an image is clearly visible
 [myBasis, EV, Scores,mask,FiltImg]=NCPCA(c,4,'Poisson',0); % Noise-corrected PCA
 Scores(:,:,2)  % Note that no residual image is visible

 Example 2 (experimental data)
 data=format_raw_data(); % loads a default dataset from the folder
 mySum=sum(dip_image(data),[],3) % displays a sumprojection of the data
 [myBasis, EV, Scores,mask,FiltImg]=NCPCA(data,4,'Poisson',10,1);

returns the Tuple (mybasis, Eigenvectors, Scores, mask_ncpca, FiltImg, NoiseCorrection)
"""
function nc_pca(mydata, max_dims=4, my_norm=norm_poisson, threshold=0, do_display=false)

    # arrange data in 3-D dataset
    # if ndims(mydata)==2;
    #     mydata=reshape(mydata, round(Int, sqrt(size(mydata,1))) ,round(Int, sqrt(size(mydata,1))),size(mydata,2))
    # end
    bg = 0

    # background estimation
    # [out,Bg,bg_im]=backgroundoffset(squeeze(mean(mydata(:,:,round(size(mydata,3)/2):size(mydata,3)-1),[],3)));

    noise_correction = my_norm(mydata)
    # @vv noise_correction

    # find peak - set all pixels below threshold to 0
    if isa(threshold, Number)
        maxv, peak_ind = findmax(mean(mydata, dims=(1,2)))
        threshold = (mydata[:,:,peak_ind[3]]) .> threshold
    end

    mask_ncpca = threshold
    # The background is removed before the normalization, however, the norm is estimated including the background
    mydata = (mydata .- bg).*mask_ncpca

    # Image is normalised by a scaling factor
    mydata_norm = mydata ./ noise_correction

    # image arranged as vectors containing each a decay.
    #allvecs = transpose(Float64.(reshape(mydata_norm,(size(mydata_norm,1)*size(mydata_norm,2), size(mydata_norm,3)))));
    allvecs = reshape(mydata_norm, (prod(size(mydata_norm)[1:2]), size(mydata_norm,3)))

    # center Data
    allvecs_cent = allvecs .- mean(allvecs, dims=1)

    # covariance matrix of dataset
    myc = (allvecs_cent' * allvecs_cent) / (size(allvecs_cent, 1) - 1)

    # SVD of covariance matrix
    U, D, V = svd(myc)

    # Extraction of first desired eigenvectors
    mybasis = V[:, 1:max_dims]

    # Optional : force positive maximum of Eigenvectors 
    for i=1:max_dims
    if abs(minimum(mybasis[:,i])) > abs(maximum(mybasis[:,i]));
        mybasis[:,i] .= .-mybasis[:,i];
    end
    end

    # calculate orthogonal projection of pixel data on the selected eigenvectors
    # = pixel scores
    # projected = transpose(transpose(mybasis)*transpose(allvecs)) 
    projected = allvecs * mybasis
    # scores=reshape(transpose(projected),((size(mydata_norm)[1:2])..., max_dims));
    scores = reshape(projected, ((size(mydata_norm)[1:2])..., max_dims));

    # Apply reverse scaling to eigenvectors           
    if prod(size(noise_correction)) > 1
        eigenvectors = mybasis .* noise_correction[1,1,:];
    else
        eigenvectors = mybasis .* noise_correction;    
    end

    # Display NC-PCA results
    if (do_display)
        plot(eigenvectors, xlabel="Time Bins", ylabel="Intensity", title="Principal Components", label=["PC1" "PC2" "PC3" "PC4"])
        @vv scores
    end

    # Filtering 
    DF=diagm(D);
    DF[max_dims+1:end, max_dims+1:end] .= 0;
    #  Forces all "unwanted" Eigenvalues to be zero and the others to be one. 
    # Thus only a projection onto the subspace is performed.
    for n=1:max_dims  
        DF[n,n] = 1;
    end

    # filtered = (U*(DF*(transpose(V)*transpose(allvecs))))
    filtered = (U*(DF*(transpose(V)*transpose(allvecs))))
    # filt_img = reshape(filtered, (size(mydata_norm,1), size(mydata_norm,2), size(filtered,1))).* noise_correction
    filt_img = reshape(filtered', size(mydata_norm)).* noise_correction
    return mybasis, eigenvectors, scores, mask_ncpca, filt_img, noise_correction
end

"""
    phasorplot(mydata, T=nothing, threshold=0, doselect=false, offsetphase=0)

This function calculates phasor coordinates of image and displays phasor plot

# INPUT ARGUMENTS
+ mydata = a noisy FLIM image (default = HeLa cell image, figure 4 from publication)
+ T = the sampling period (e.g 50 ns)
+ threshold = minimum intensity in peak of decay (default = 0)
+ doselect = select ROI from image before phasor analysis (default = 0)
+ offsetphase = compensate for wide IRF by starting projection at later bins  (default = 0)
+ after peak - indicate number of bins
 
# OUTPUT ARGUMENTS
+ myhist = histogram of phasor coordinates in phasor space
+ [realPart,imagPart] = images with real and imaginary phasor coordinates
+ mask_phasor = mask of selected pixels after threshold 
+ ROI = mask of selected pixels after ROI selection

Please refer to and cite where appropriate: "Noise-corrected principal
component analysis of fluorescence lifetime imaging (FLIM) data" (2016)
by Alix Le Marois, Simon Labouesse, Klaus Suhling, and Rainer Heintzmann

"""
function phasorplot(mydata, T=nothing, threshold=0, doselect=false, offsetphase=0)
            
    if ndims(mydata)==3;
        nbtimebins=size(mydata,3);
    else
        nbtimebins=size(mydata,2); 
        nbpixels=size(mydata,1)
        mydata=reshape(mydata, round(Int,sqrt(nbpixels)), round(Int,sqrt(nbpixels)), nbtimebins);
    end
    
    bg = 0

    if isnothing(T)
        T=nbtimebins;
    end
        
    # average frame
    projM = mean(mydata, dims=3);
    # average decay
    projMS = mean(mydata, dims=(1, 2));
    # size(projMS)
    # determine peak and end of decay
    maxv, peak_ind = findmax(projMS)
    peak_ind = peak_ind[3]
    
    # max sub-threshold pixels
    mask_phasor = projM .> threshold
    
    # background estimation from end half of decay in empty region
    # out,bg,bg_im = backgroundoffset(squeeze(mean(mydata(:,:,round(size(mydata,3)/2):size(mydata,3)-1),[],3)));

    # select image ROI
    # if doselect
    #     M_int=double(sum(mydata,[],3));
    #     image(M_int);
    #     #  colormap(jet);
    #     title("intensity image");
    #     caxis([min(min(M_int)) max(max(M_int))]);       
    #     ROI=roipoly;      
    #     mydata=mydata.*ROI;
    # end

    Mbgc = mydata .- bg;
    
    # determine end of decay
    @show end_ind = findlast(projMS .> bg)[3]
    
    # set frequency to decay size and phase to decay peak
    myramp = range(0,T,nbtimebins)
    omega = 2pi / ((end_ind - peak_ind + 1)*T/nbtimebins)
    phase = omega*(peak_ind+offsetphase)*T/nbtimebins
    
    # cos and sin matrices
    # easy way, but has troubles with sampling issues
    cosmat = reshape(cos.(omega .* myramp[peak_ind:end_ind] .- phase), (1, 1, end_ind-peak_ind+1))
    sinmat = reshape(sin.(omega .* myramp[peak_ind:end_ind] .- phase), (1, 1, end_ind-peak_ind+1))
    
    # calculate summed intensity for each pixel
    sumAll=sum(Mbgc[:,:,peak_ind:end_ind],dims=3)
    
    # project each decay on cos and sin matrices
    realPart = sum(cosmat .* Mbgc[:,:,peak_ind:end_ind], dims=3)./sumAll
    imagPart = sum(sinmat .* Mbgc[:,:,peak_ind:end_ind], dims=3)./sumAll

    @show sum(mask_phasor)
    bad_pixels = .!mask_phasor .|| isnan.(realPart) .|| isnan.(imagPart) .|| (realPart .== 0 .&& imagPart .== 0)
    realPart = realPart[.!bad_pixels]
    imagPart = imagPart[.!bad_pixels] 
    
    # create histogram of phasor coordinates
    hp = histogram2d(realPart[:], imagPart[:],
                    bins = 200, xrange=[-0.1,1.1], yrange=[-0.1, 0.6],
                    c=cgrad([:white,:red, :blue],[0.01, 0.3, 0.8])) # mask_phasor,[sx,ex],[sy,ey],nx,ny);
    # hp = histogram2d(realPart[:], imagPart[:], bins=200) # mask_phasor,[sx,ex],[sy,ey],nx,ny);
    
    # plot universal circle
    # figure
    x = range(0,1,100);
    circ = sqrt.(x-x.*x) ;
    plot!(x, circ, xlabel="Real Part", label=false,
            ylabel="Imaginary Part", title="Phasor Plot");
    display(hp)
    
    # plot transforms of pixels
    # heatmap(sx.+(ex.-sx)*[0:nx]/nx, sy+(ey-sy).*[0:ny]/ny,(double(myhist./max(myhist)*64)))
    # colormap('parula')
    # colormap("default")
    #plot!(x, circ)
    
    # plot representative decay + frequency adapted sin and cos waves
    if (false)
        dat = mean(Mbgc, dims=(1,2))[:];
        p = plot(myramp, dat ./ maximum(dat), title="representative decay with cos and sin modulations", xlabel="Time Bins", ylabel="Normalized Intensity", label="Average decay")
        plot!(myramp[peak_ind:end_ind], mean(cosmat, dims=(1,2))[:], label="cosine modulation")
        plot!(myramp[peak_ind:end_ind], mean(sinmat,dims=(1,2))[:], label="sine modulation")
        display(p)
    end

    return hp, realPart,imagPart,mask_phasor
end  
    