#!/bin/python3

import os
import sys
import cv2
import gc
import numpy as np

from scipy.optimize import curve_fit
from obspy import UTCDateTime
from pandas import DataFrame, date_range
from tqdm import tqdm
from pandas import read_pickle

import matplotlib.pyplot as plt
import matplotlib as mpl

if os.uname().nodename == 'lighthouse':
    root_path = '/home/andbro/'
    data_path = '/home/andbro/kilauea-data/'
    archive_path = '/home/andbro/freenas/'
    bay_path = '/home/andbro/ontap-ffb-bay200/'
elif os.uname().nodename == 'kilauea':
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'
elif os.uname().nodename in ['lin-ffb-01', 'ambrym', 'hochfelln']:
    root_path = '/home/brotzer/'
    data_path = '/import/kilauea-data/'
    archive_path = '/import/freenas-ffb-01-data/'
    bay_path = '/import/ontap-ffb-bay200/'


## configuration
config = {}

## pass camera name
config['camera'] = sys.argv[1]

## specify if checkup figures are stored
config['store_figures'] = True

config['path_to_data'] = archive_path+f"ids/images{config['camera']}/"
#config['path_to_data'] = input("Enter path to images: ")

config['path_to_outdata'] = data_path+f"ids/data{config['camera']}/"

if len(sys.argv) > 2:
    config['date1'] = str(UTCDateTime(sys.argv[2]).date).replace("-","")
else:
    config['date1'] = str((UTCDateTime().now() -86400).date).replace("-","")
#config['date1'] = input("Enter Date1: ")

config['nth'] = 10
#config['nth'] = int(input("Every nth images [1]: ")) or 1

# define initial guess for different camera setups [ amplitude, xo, yo, sigma_x, sigma_y, theta, offset ]
if UTCDateTime(config['date1']) >= UTCDateTime("2024-08-01"):
    # 2024-08-01
    config['initial_guess0'] = {"00": [255, 600, 600, 500, 500, 0, 0],
                                "01": [255, 550, 550, 500, 500, 0, 0],
                                "03": [255, 500, 450, 500, 500, 0, 0],
                                "05": [255, 550, 500, 500, 500, 0, 0],
                                "07": [255, 650, 400, 500, 500, 0, 0],
                            }
elif UTCDateTime(config['date1']) >= UTCDateTime("2024-07-29"):
    # 2024-07-29
    config['initial_guess0'] = {"00": [255, 2000, 1000, 500, 500, 0, 0],
                                "01": [255, 550, 550, 500, 500, 0, 0],
                                "03": [255, 700, 500, 500, 500, 0, 0],
                                "07": [255, 750, 500, 500, 500, 0, 0],
                            }

elif UTCDateTime(config['date1']) <= UTCDateTime("2024-04-01"):
    # 2024-03-01 - 2024-04-01
    config['initial_guess0'] = {"00": [255, 1700, 950, 500, 500, 0, 0],
                            }
else:
    config['initial_guess0'] = {"00": [255, 2400, 500, 500, 500, 0, 0],
                                "01": [255, 700, 550, 500, 500, 0, 0],
                                "03": [255, 780, 550, 500, 500, 0, 0],
                                "07": [255, 750, 550, 500, 500, 0, 0],
                            }

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):

    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    return g.ravel()

def __makeplot(im, im_fitted, x_max, y_max, x, y, amax=255):

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    maxY, maxX = im.shape

    font = 12

    Nrow, Ncol = 4, 6

    fig = plt.figure(figsize=(15, 8))

    gs = GridSpec(Nrow, Ncol, figure=fig, hspace=0.5, wspace=0.45)

    ax1 = fig.add_subplot(gs[0:4, :4])
    ax2 = fig.add_subplot(gs[0:2, 4:])
    ax3 = fig.add_subplot(gs[2:4, 4:])

    if amax > 255:
        ax1.imshow(im, cmap=plt.get_cmap("gray"), vmax=255);
    else:
        ax1.imshow(im, cmap=plt.get_cmap("gray"), vmax=amax);
    ax1.contour(x, y, im_fitted);
    ax1.scatter(x_max, y_max, color="red", marker="d");

    ax1.axvline(x_max, 0, maxY, color="tab:orange", ls=":", zorder=1, alpha=0.4)
    ax1.axhline(y_max, 0, maxX, color="tab:blue", ls=":", zorder=1, alpha=0.4)

    ax2.plot(im[y_max, :], color="tab:blue", zorder=2)
    ax2.plot(im_fitted[y_max, :], color="black", ls="--", zorder=2)
    ax2.axvline(x_max, 0, im_fitted[y_max, x_max], color="red", ls="--", zorder=1, label=f"X = {x_max}")

    ax3.plot(im[:, x_max], color="tab:orange", zorder=2)
    ax3.plot(im_fitted[:, x_max], color="black", ls="--", zorder=2)
    ax3.axvline(y_max, 0, im_fitted[y_max, x_max], color="red", ls="--", zorder=1, label=f"Y = {y_max}")


    ax1.set_xlabel("Pixel X", fontsize=font)
    ax1.set_ylabel("Pixel Y", fontsize=font)
    ax2.set_xlabel("Pixel X", fontsize=font)
    ax3.set_xlabel("Pixel Y", fontsize=font)

    ax2.set_ylabel("Intensity", fontsize=font)
    ax3.set_ylabel("Intensity", fontsize=font)

    ax2.set_ylim(0, 255)
    ax3.set_ylim(0, 255)

    ax2.legend()
    ax3.legend()

    ax1.set_ylim(0, maxY)
    ax1.set_xlim(0, maxX)
    ax2.set_xlim(0, maxX)
    ax3.set_xlim(0, maxY)

    ax2.grid(zorder=0, color="grey", ls="--", alpha=0.5)
    ax3.grid(zorder=0, color="grey", ls="--", alpha=0.5)

    ax1.text(0.01, 0.98, "(a)", ha="left", va="top", transform=ax1.transAxes, fontsize=font+1, color="w")
    ax2.text(0.01, 0.98, "(b)", ha="left", va="top", transform=ax2.transAxes, fontsize=font+1, color="k")
    ax3.text(0.01, 0.98, "(c)", ha="left", va="top", transform=ax3.transAxes, fontsize=font+1, color="k")


    #plt.show();
    return fig

def __store_as_pickle(obj, filename):

    import pickle
    from os.path import isdir

    ofile = open(filename, 'wb')
    pickle.dump(obj, ofile)

    if isdir(filename):
        print(f"created: {filename}")

def main():

    # initialize
    config['initial_guess'] = config['initial_guess0']

    # run loop over dates
    for _date in date_range(config['date1'], config['date1']):

        # get date as str
        date_str = str(_date)[:10]

        path_to_data = f"{config['path_to_data']}{date_str.replace('-','')}/"

        # read files in directory
        try:
            files = os.listdir(path_to_data)
        except:
            print(f" -> {date_str} not in directory!")
            continue

        # dummy data
        dummy = np.ones(len(files[::config['nth']]))*np.nan

        # prepare output dataframe
        df_out = DataFrame()

        for col in ["time", "x", "y", "x_idx", "y_idx", "amp", "x_sig", "y_sig", "theta", "offset", "x_var", "y_var", "amp_var", "y_sig_var", "x_sig_var", "theta_var", "offset_var"]:
            df_out[col] = dummy

        for _n, file in enumerate(tqdm(files[::config['nth']])):

            config['initial_guess'][config['camera']] = config['initial_guess0'][config['camera']]

            # load last estimate as intial guess
            try:

                # read file
                guess = read_pickle(config['path_to_outdata']+"tmp/"+f"{config['camera']}_initial_guess.pkl")

                # check for integrity
                if not np.isnan(guess).any() and not np.isinf(guess).any() and guess[0] > 0:
                    if guess[1] > 0 and guess[2] > 0:
                        config['initial_guess'][config['camera']][1] = int(guess[1])
                        config['initial_guess'][config['camera']][2] = int(guess[2])

            except Exception as e:
                print(e)
                print(f" -> failed to load initial guess")


            # check data type
            if file[-3:] != "png":
                print(f" -> not a png: {file}")
                continue

            # load image and get dimensions
            try:

                # load image
                im = cv2.imread(path_to_data+file, -1)

                # flip vertically
                # im = np.array(list(reversed(im)))

                # flip horizontally
                im = np.array([list(reversed(row)) for row in im])

                # obtain dimensions
                h, w = im.shape

                # create data array
                data = im.ravel()

            except:
                print(f" -> failed to load image: {file}")
                continue

            # check for maximum amplitude (= dark images)
            if max(im.reshape(h*w, 1)) < 10:
                print(f" -> image dark! stop!")
                continue

            # prepare x-y-mesh
            x = np.linspace(0, w, w)
            y = np.linspace(0, h, h)
            x, y = np.meshgrid(x, y)

            # initial guess of parameters [ amplitude, xo, yo, sigma_x, sigma_y, theta, offset ]
            initial_guess = config['initial_guess'][config['camera']]

            try:
                # find the optimal Gaussian parameters
                popt, pcov = curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess,
                                       bounds=([0, 0, 0, 0, 0, 0, -np.inf],
                                               [1000, w, h, 2000, 2000, 360, np.inf])
                                      )

                # create new data with these parameters
                data_fitted = twoD_Gaussian((x, y), *popt)

            except:
                # give it a go with the manual set values
                try:
                    # initial guess of parameters [ amplitude, xo, yo, sigma_x, sigma_y, theta, offset ]
                    initial_guess = config['initial_guess0'][config['camera']]

                    # find the optimal Gaussian parameters
                    popt, pcov = curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess)

                    # create new data with these parameters
                    data_fitted = twoD_Gaussian((x, y), *popt)
                except:
                    print(f" -> estimation failed !")
                    continue

#            print(popt)

            # get diagonal values
            pcov_diag = np.diag(pcov)

            # reshape to image dimensions
            im_fitted = data_fitted.reshape(h, w)

            # get maximum of 2d fit
            y_max, x_max = np.argwhere(im_fitted == im_fitted.max())[0]
            print(f" -> X: {x_max}  Y: {y_max}")

            date_str = file.split('.')[0].split('_')[0]
            time_str = file.split('.')[0].split('_')[1]

            df_out.loc[_n, 'time'] = str(UTCDateTime(date_str+"T"+time_str))

            df_out.loc[_n, 'y_idx'] = y_max
            df_out.loc[_n, 'x_idx'] = x_max

            df_out.loc[_n, 'amp'] = popt[0]
            df_out.loc[_n, 'x'] = popt[1]
            df_out.loc[_n, 'y'] = popt[2]
            df_out.loc[_n, 'x_sig'] = popt[3]
            df_out.loc[_n, 'y_sig'] = popt[4]
            df_out.loc[_n, 'theta'] = popt[5]
            df_out.loc[_n, 'offset'] = popt[6]

            df_out.loc[_n, 'amp_var'] = pcov_diag[0]
            df_out.loc[_n, 'x_var'] = pcov_diag[1]
            df_out.loc[_n, 'y_var'] = pcov_diag[2]
            df_out.loc[_n, 'x_sig_var'] = pcov_diag[3]
            df_out.loc[_n, 'y_sig_var'] = pcov_diag[4]
            df_out.loc[_n, 'theta_var'] = pcov_diag[5]
            df_out.loc[_n, 'offset_var'] = pcov_diag[6]

            # make tmp directory if not existent
            if not os.path.isdir(config['path_to_outdata']+"tmp/"):
                os.mkdir(config['path_to_outdata']+"tmp/")

            # write current maximum estimate to file for next inital guess
            __store_as_pickle(popt, config['path_to_outdata']+"tmp/"+f"{config['camera']}_initial_guess.pkl")


            if _n % 10 == 0:
                mpl.use('Agg')

                #fig = plt.figure();
                #plt.imshow(im, cmap=plt.get_cmap("gray"));
                #plt.contour(x, y, im_fitted);
                #plt.scatter(x_max, y_max, color="red", marker="d");

                if config['store_figures']:
                    if not os.path.isdir(config['path_to_outdata']+"outfigs/"):
                        os.mkdir(config['path_to_outdata']+"outfigs/")
                    try:
                        fig = __makeplot(im, im_fitted, x_max, y_max, x, y, amax=popt[0]*10)
                        fig.savefig(config['path_to_outdata']+"outfigs/"+f"{file[:-4]}_mod.png", format="png", dpi=150, bbox_inches='tight');
                    except Exception as e:
                        print(e)

            _gc = gc.collect();

        # sort for time column
        df_out.sort_values(by="time", inplace=True)

        # write output data frame
        print(f" -> write output data {config['path_to_outdata']}{date_str}.pkl")
        df_out.to_pickle(config['path_to_outdata']+f"{date_str}.pkl")



if __name__ == "__main__":
    main()

## End of File
