import os
import argparse
import glob
from datetime import datetime

from models.dncnn import DnCNN, DnCNN_c, Estimation_direct
from models.cbdnet import CBDNet
from utils import *

parser = argparse.ArgumentParser(description="PD-denoising")

# model parameter
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--delog", type=str, default="logs", help='path of log and model files')
parser.add_argument("--mode", type=str, default="M", help='CBDNet (CBD) or DnCNN-B (B) or MC-AWGN-RVIN (MC)')

# tested noise type
parser.add_argument("--color", type=int, default=0, help='[0]gray [1]color')
parser.add_argument("--real_n", type=int, default=0, help='real noise or synthesis noise [0]synthetic noises [1]real noisy image wo gnd [2]real noisy image with gnd')
parser.add_argument("--spat_n", type=int, default=0, help='whether to add spatial-variant signal-dependent noise, [0]no spatial [1]Gaussian-possion noise')

# pixel-shuffling parameter
parser.add_argument("--ps", type=int, default=0, help='pixel shuffle [0]no pixel-shuffle [1]adaptive pixel-ps [2]pre-set stride')
parser.add_argument("--ps_scale", type=int, default=2, help='if ps==2, use this pixel shuffle stride')

# down-scaling parameter
parser.add_argument("--scale", type=float, default=1, help='resize the original images')
parser.add_argument("--rescale", type=int, default=1, help='resize it back to the origianl size after downsampling')

# testing data path and processing
parser.add_argument("--test_data", type=str, default='Set12', help='testing data path')
parser.add_argument("--test_data_gnd", type=str, default='Set12', help='testing data ground truth path if it exists')
parser.add_argument("--cond", type=int, default=1, help='Testing mode using noise map of: [0]Groundtruth [1]Estimated [2]External Input')
parser.add_argument("--test_noise_level", nargs = "+",  type=int, help='input noise level while generating noisy images')
parser.add_argument("--ext_test_noise_level", nargs = "+", type=int, help='external noise level input used if cond==2')

# refining on the estimated noise map
parser.add_argument("--refine", type=int, default=0, help='[0]no refinement of estimation [1]refinement of the estimation')
parser.add_argument("--refine_opt", type=int, default=0, help='[0]get the most frequent [1]the maximum [2]Gaussian smooth [3]average value of 0 and 1 opt')
parser.add_argument("--zeroout", type=int, default=0, help='[0]no zeroing out [1]zeroing out some maps')
parser.add_argument("--keep_ind", nargs = "+", type=int, help='[0 1 2]Gaussian [3 4 5]Impulse')

# output options
parser.add_argument("--output_map", type=int, default=0, help='whether to output maps')
parser.add_argument("--k", type=float, default=1, help='merging factor between details and background')
parser.add_argument("--out_dir", type=str, default="results_bc", help='path of output files')

# New options
parser.add_argument("--pth_dir", type=str, default="results_bc", help='path of output files')

opt = parser.parse_args()

# the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0, 75], [0, 80]]


def img_normalize(data):
    return data/255.


def main():
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    # Build model
    net = None
    c = 1 if opt.color == 0 else 3
    if opt.mode == "MC":
        net = DnCNN_c(channels=c, num_of_layers=opt.num_of_layers, num_of_est=2 * c)
        est_net = Estimation_direct(c, 2 * c)
    elif opt.mode == "B":
        net = DnCNN(channels=c, num_of_layers=opt.num_of_layers)
    elif opt.mode == 'CBD':
        net = CBDNet(channels=3)

    print('Loading the model...\n')
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids)
    model_info = torch.load(os.path.join(opt.delog, 'cbdnet_syn.pth.tar'),  map_location=torch.device('cpu'))
    model.load_state_dict(model_info['state_dict'])
    # model.load_state_dict(model_info)
    model.eval()

    #Estimator Model
    if opt.mode == "MC":
        model_est = nn.DataParallel(est_net, device_ids=device_ids)
        model_est.load_state_dict(torch.load(os.path.join(opt.delog, 'logs_color_MC_AWGN_RVIN/est_net.pth'), map_location=torch.device('cpu')))
        model_est.eval()
    elif opt.mode == 'CBD':
        model_est = net.fcn

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('../../data', opt.test_data, '*.*'))
    files_source.sort()
    # process data

    #for the input condition modeling
    noise_level_list = np.zeros((2 * c,1))  #two noise types with three channels
    if opt.cond == 0:  #if we use the ground truth of noise for denoising, and only one single noise type
        noise_level_list = np.array(opt.test_noise_level)
        print(noise_level_list)
    elif opt.cond == 2:  #if we use an external fixed input condition for denoising
        noise_level_list = np.array(opt.ext_test_noise_level)

    # process images with pre-defined noise level
    psnr_test = 0
    for f in files_source:
        print(datetime.now(), f)
        file_name = f.split('/')[-1].split('.')[0]
        if opt.real_n == 2:  #have ground truth
            gnd_file_path = os.path.join('../../data', opt.test_data_gnd, f.split('/')[-1])
            print(gnd_file_path)
            Img_gnd = cv2.imread(gnd_file_path)
            Img_gnd = Img_gnd[:,:,::-1]
            Img_gnd = cv2.resize(Img_gnd, (0,0), fx=opt.scale, fy=opt.scale)
            Img_gnd = img_normalize(np.float32(Img_gnd))


        # image
        Img = cv2.imread(f)  #input image with w*h*c
        w, h, _ = Img.shape
        Img = Img[:,:,::-1]  #change it to RGB
        Img = cv2.resize(Img, (0,0), fx=opt.scale, fy=opt.scale)
        if opt.color == 0:
            Img = Img[:,:,0]  #For gray images
            Img = np.expand_dims(Img, 2)

        original_noisy_image = torch.clamp(np2ts(img_normalize(np.float32(Img)), opt.color), 0., 1.)

        # if opt.mode == 'CBD':
        #     nu_noise_level, output_image = model(original_noisy_image)
        #     output_image = torch.clamp(output_image, 0., 1.)
        # else:
        #     print('TEST')
        #     noise_estimation = torch.clamp(model_est(original_noisy_image), 0., 1.)
        #     noise_estimation_dncnn = model(original_noisy_image, noise_estimation)
        #     output_image = torch.clamp(original_noisy_image - noise_estimation_dncnn, 0., 1.)
        #
        # output_image = output_image.data.cpu().numpy()
        # output_image = np.mean(output_image, 0)
        # output_image = np.transpose(output_image, (1, 2, 0)) * 255.0
        #
        # gt_image_tensor = np2ts(Img_gnd)
        #
        # output_image = output_image.astype(np.float32)
        #
        # cv2.imwrite(os.path.join(opt.out_dir, file_name + '.png'), output_image[:, :, ::-1])
        #
        # output_image = np2ts(output_image / 255.)
        #
        # if opt.real_n == 0 or opt.real_n == 2:
        #     psnr = batch_PSNR(output_image, gt_image_tensor, 1.)
        #     psnr_test += psnr
        #     print("%s PSNR %f\n" % (f, psnr))
        #
        # continue

        pss=1
        if opt.ps == 1:
            pss = decide_scale_factor(Img/255., model_est, color=opt.color,  thre = 0.008, plot_flag = 1, stopping = 4,mark = opt.out_dir + '/' +  file_name)[0]
            print(pss)
            Img = pixelshuffle(Img, pss)
        elif opt.ps == 2:
            pss = opt.ps_scale
            Img = pixelshuffle(Img, pss)
        Img = img_normalize(np.float32(Img))

        #Clean Image Tensor for evaluation
        if opt.real_n == 2:
            ISource = np2ts(Img_gnd)
        else:
            ISource = np2ts(Img)
        # noisy image and true residual
        if opt.real_n == 0 and opt.spat_n == 0:  #no spatial noise setting, and synthetic noise
            noisy_img = generate_comp_noisy(Img, np.array(opt.test_noise_level) / 255.)
            if opt.color == 0:
                noisy_img = np.expand_dims(noisy_img[:,:,0], 2)
        elif opt.real_n == 1 or opt.real_n == 2:  #testing real noisy images
            noisy_img = Img
        elif opt.spat_n == 1:
            noisy_img = generate_noisy(Img, 2, 0, 20, 40)
        INoisy = np2ts(noisy_img, opt.color)
        INoisy = torch.clamp(INoisy, 0., 1.)
        True_Res = INoisy - ISource
        with torch.no_grad():
            ISource, INoisy, True_Res, original_noisy_image = Variable(ISource), Variable(INoisy), Variable(True_Res), Variable(original_noisy_image)

        if opt.mode == 'CBD':
            # obtain the corresponding input_map
            if opt.cond == 0 or opt.cond == 2:  # if we use ground choose level or the given fixed level

                # normalize noise leve map to [0,1]
                noise_level_list_n = np.zeros((2 * c, 1))
                for noise_type in range(2):
                    for chn in range(c):
                        noise_level_list_n[noise_type * c + chn] = normalize(
                            a=noise_level_list[noise_type * 3 + chn],
                            len_v=1,
                            min_v=limit_set[noise_type][0],
                            max_v=limit_set[noise_type][1]
                        )

                # generate noise maps
                noise_map = np.zeros((1, 2 * c, Img.shape[0], Img.shape[1]))  # initialize the noise map
                noise_map[0, :, :, :] = np.reshape(
                    np.tile(noise_level_list_n, Img.shape[0] * Img.shape[1]), (2 * c, Img.shape[0], Img.shape[1])
                )
                NM_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
                with torch.no_grad():
                    NM_tensor = Variable(NM_tensor)

            # use the estimated noise-level map for blind denoising
            elif opt.cond == 1:  # if we use the estimated map directly
                NM_tensor = torch.clamp(model_est(INoisy), 0., 1.)
                if opt.refine == 1:  # if we need to refine the map before putting it to the denoiser
                    # refine_opt can be max, freq and their average
                    NM_tensor_bundle = level_refine(NM_tensor, opt.refine_opt, 2 * c)
                    NM_tensor = NM_tensor_bundle[0]
                    noise_estimation_table = np.reshape(NM_tensor_bundle[1], (2 * c,))
                if opt.zeroout == 1:
                    NM_tensor = zeroing_out_maps(NM_tensor, opt.keep_ind)

            NU_noise_level, Res = model(INoisy, NM_tensor)
            Out = torch.clamp(Res, 0., 1.)  # Output image after denoising

        if opt.mode == "MC":
            # obtain the corresponding input_map
            if opt.cond == 0 or opt.cond == 2:  #if we use ground choose level or the given fixed level
                #normalize noise leve map to [0,1]
                noise_level_list_n = np.zeros((2*c, 1))
                print(c)
                for noise_type in range(2):
                    for chn in range(c):
                        noise_level_list_n[noise_type * c + chn] = normalize(noise_level_list[noise_type * 3 + chn], 1, limit_set[noise_type][0], limit_set[noise_type][1])  #normalize the level value
                #generate noise maps
                noise_map = np.zeros((1, 2 * c, Img.shape[0], Img.shape[1]))  #initialize the noise map
                noise_map[0, :, :, :] = np.reshape(np.tile(noise_level_list_n, Img.shape[0] * Img.shape[1]), (2*c, Img.shape[0], Img.shape[1]))
                NM_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
                with torch.no_grad():
                    NM_tensor = Variable(NM_tensor)

            #use the estimated noise-level map for blind denoising
            elif opt.cond == 1:  #if we use the estimated map directly
                NM_tensor = torch.clamp(model_est(INoisy), 0., 1.)
                if opt.refine == 1:  #if we need to refine the map before putting it to the denoiser
                    NM_tensor_bundle = level_refine(NM_tensor, opt.refine_opt, 2*c)  #refine_opt can be max, freq and their average
                    NM_tensor = NM_tensor_bundle[0]
                    noise_estimation_table = np.reshape(NM_tensor_bundle[1], (2 * c,))
                if opt.zeroout == 1:
                    NM_tensor = zeroing_out_maps(NM_tensor, opt.keep_ind)
            Res = model(INoisy, NM_tensor)
            Out = torch.clamp(INoisy - Res, 0., 1.)  # Output image after denoising

        elif opt.mode == "B":
            Res = model(INoisy)
            Out = torch.clamp(INoisy-Res, 0., 1.)  #Output image after denoising

        #get the maximum denoising result
        max_NM_tensor = level_refine(NM_tensor, 1, NM_tensor.shape[1])[0]
        NU_noise_level, max_Res = model(original_noisy_image, max_NM_tensor)
        if opt.mode == 'CBD':
            max_Out = torch.clamp(max_Res, 0., 1.)
        else:
            max_Out = torch.clamp(original_noisy_image - max_Res, 0., 1.)

        max_out_numpy = max_Out.data.cpu().numpy()
        max_out_numpy = np.mean(max_out_numpy, 0)
        max_out_numpy = np.transpose(max_out_numpy, (1, 2, 0)) * 255.0

        # max_out_numpy = visual_va2np(max_Out, opt.color, opt.ps, pss, 1, opt.rescale, w, h, c)
        # max_out_numpy = visual_va2np(max_Out, opt.color, 1, 1, 1, opt.rescale, w, h, c)
        del max_Out
        del max_Res
        del max_NM_tensor

        if (opt.ps == 1 or opt.ps == 2) and pss!=1:  #pixelshuffle multi-scale
            #create batch of images with one subsitution
            mosaic_den = visual_va2np(Out, opt.color, 1, pss, 1, opt.rescale, w, h, c)
            out_numpy = np.zeros((pss ** 2, c, w, h))
            #compute all the images in the ps scale set
            for row in range(pss):
                for column in range(pss):
                    re_test = visual_va2np(Out, opt.color, 1, pss, 1, opt.rescale, w, h, c, 1, visual_va2np(INoisy, opt.color), [row, column])/255.
                    #cv2.imwrite(os.path.join(opt.out_dir,file_name + '_%d_%d.png' % (row, column)), re_test[:,:,::-1]*255.)
                    re_test = np.expand_dims(re_test, 0)
                    if opt.color == 0:  #if gray image
                        re_test = np.expand_dims(re_test[:, :, :, 0], 3)
                    re_test_tensor = torch.from_numpy(np.transpose(re_test, (0,3,1,2))).type(torch.FloatTensor)
                    with torch.no_grad():
                        re_test_tensor = Variable(re_test_tensor)
                    re_NM_tensor = torch.clamp(model_est(re_test_tensor), 0., 1.)
                    if opt.refine == 1:  #if we need to refine the map before putting it to the denoiser
                            re_NM_tensor_bundle = level_refine(re_NM_tensor, opt.refine_opt, 2*c)  #refine_opt can be max, freq and their average
                            re_NM_tensor = re_NM_tensor_bundle[0]
                    NU_noise_level, re_Res = model(re_test_tensor, re_NM_tensor)
                    if opt.mode == 'CBD':
                        Out2 = torch.clamp(re_Res, 0., 1.)
                    else:
                        Out2 = torch.clamp(re_test_tensor - re_Res, 0., 1.)
                    out_numpy[row*pss+column,:,:,:] = Out2.data.cpu().numpy()
                    del Out2
                    del re_Res
                    del re_test_tensor
                    del re_NM_tensor
                    del re_test

            out_numpy = np.mean(out_numpy, 0)
            out_numpy = np.transpose(out_numpy, (1,2,0)) * 255.0
        elif opt.ps == 0 or pss==1:  #other cases
            out_numpy = visual_va2np(Out, opt.color, 0, 1, 1, opt.rescale, w, h, c)

        out_numpy = out_numpy.astype(np.float32)  #details
        max_out_numpy = max_out_numpy.astype(np.float32)  #background

        # Merging the texture details and flat regions to balance the effect
        k = opt.k
        merge_out_numpy = (1-k)*out_numpy + k*max_out_numpy
        merge_out_numpy = merge_out_numpy.astype(np.float32)

        # Saving the resultant image
        cv2.imwrite(os.path.join(opt.out_dir, file_name + '_pss'+str(pss)+'_k'+str(k)+'.png'), merge_out_numpy[:,:,::-1])

        NM_Out = np2ts(merge_out_numpy/255.)

        # Calculate the batch PSNR
        if opt.real_n == 0 or opt.real_n == 2:
            psnr = batch_PSNR(NM_Out, ISource, 1.)
            psnr_test += psnr
            print("%s PSNR %f\n" % (f, psnr))

    # Calculate the average PSNR
    if opt.real_n == 0 or opt.real_n == 2:
        psnr_test /= len(files_source)
        print("\nPSNR on test data %f" % psnr_test)


if __name__ == "__main__":
    main()
