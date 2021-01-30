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

        if opt.mode == 'CBD':
            nu_noise_level, output_image = model(original_noisy_image)
            output_image = torch.clamp(output_image, 0., 1.)
        else:
            print('TEST')
            noise_estimation = torch.clamp(model_est(original_noisy_image), 0., 1.)
            noise_estimation_dncnn = model(original_noisy_image, noise_estimation)
            output_image = torch.clamp(original_noisy_image - noise_estimation_dncnn, 0., 1.)

        output_image = output_image.data.cpu().numpy()
        output_image = np.mean(output_image, 0)
        output_image = np.transpose(output_image, (1, 2, 0)) * 255.0

        gt_image_tensor = np2ts(Img_gnd)

        output_image = output_image.astype(np.float32)

        cv2.imwrite(os.path.join(opt.out_dir, file_name + '.png'), output_image[:, :, ::-1])

        output_image = np2ts(output_image / 255.)

        # Calculate the batch PSNR
        if opt.real_n == 0 or opt.real_n == 2:
            psnr = batch_PSNR(output_image, gt_image_tensor, 1.)
            psnr_test += psnr
            print("%s PSNR %f\n" % (f, psnr))

    # Calculate the average PSNR
    if opt.real_n == 0 or opt.real_n == 2:
        psnr_test /= len(files_source)
        print("\nPSNR on test data %f" % psnr_test)


if __name__ == "__main__":
    main()
