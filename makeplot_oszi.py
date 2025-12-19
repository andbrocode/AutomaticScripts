#!/usr/bin/python3

"""
Spyder Editor

This is a script to capture the screen of RIGOL DS1054Z Oscilloscope

Based on python3


__author__ = 'AndBro'
__year__   = '2022'

"""

# _____________________________________________________________
'''---- import libraries ----'''

from ds1054z import DS1054Z
from IPython.display import Image, display
from datetime import datetime
from numpy import asarray
from PIL import Image, ImageFile

import os
os.environ['MPLCONFIGDIR'] = "/import/kilauea-data/tmp_cache_matplotlib/"

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# _____________________________________________________________
''' ---- configurations ---- '''

# define scope IP address
DS1054Z_IP = "10.153.82.107"

# specify path to save images
path_to_images = "/home/brotzer/Downloads/"


# oscilloscope settings
#channel1_volt = 1.0  # in volts
#time_scaling = 0.002 # in seconds

file = path_to_images+"tmp.bmp"

# _____________________________________________________________
'''---- MAIN ----'''

def main():

    dt = datetime.utcnow()

    # initiate scope object
    try:
        scope = DS1054Z(DS1054Z_IP)
        # scope = DS1054Z(f'TCPIP::{DS1054Z_IP}::INSTR')
    except:
        return

    try:
        #stop scope
        scope.stop()

        scope.set_probe_ratio(1, 1) ## set screen ratio

        # take a screenshot
        # print(" -> taking screenshot...")
        bmap_scope = scope.display_data

        try:
            # save image to file
            with open(file, "wb") as _img:
                _img.write(bmap_scope)
           
        except:
            print(f" -> failed to store image")
            return

        # restart scope
        scope.run()
    except:
        print(f"failed to get screenshot")
        return

    # display the screentshot
    #display(Image(bmap_scope))
    
#    print(type(bmap_scope))

    try:
        # load image
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(file).convert('RGBA')
#        img = mpimg.imread(file)

    except Exception as e:
        print(e)
        print(" -> failed to load image")
        return
    
    try:
        # plot image as figure with time stamp
        fig = plt.figure()
        plt.title(str(dt)[:19]+" UTC")
        plt.imshow(img)
        plt.axis("off")
        plt.close();

        # save image with time stamp
        fig.savefig(path_to_images+f"html_oszi.png", format="png", dpi=150, bbox_inches='tight');

    except Exception as e:
        print(f" -> failed to save figure")
        print(e)
        pass

if __name__ == "__main__":
    main()

# End of File
