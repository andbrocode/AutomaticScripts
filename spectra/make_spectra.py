# %% [markdown]

# Make Spectra of station

# Using YAML config
# python make_spectra.py -c config.yml

# %%

import os
import gc
import sys
import json
import yaml
import argparse
from obspy import UTCDateTime
from spectra import spectra


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make spectra for a station')
    parser.add_argument('-c', '--config', type=str, required=True,
                      help='Path to configuration file (JSON or YAML)')
    parser.add_argument('-d', '--date', type=str,
                      help='Process specific date (YYYY-MM-DD). Overrides config dates')
    return parser.parse_args()

def load_config(config_path):
    """
    Load configuration from JSON or YAML file.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file type from extension
    ext = os.path.splitext(config_path)[1].lower()
    
    try:
        with open(config_path, 'r') as f:
            if ext == '.json':
                config = json.load(f)
            elif ext in ['.yml', '.yaml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
    except Exception as e:
        raise Exception(f"Error loading configuration file: {e}")
    
    return config

def main():
    """Main function to process spectra."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)

    # Override dates if date argument is provided
    if args.date:
        try:
            date = UTCDateTime(args.date)
            config['tbeg'] = date.strftime("%Y-%m-%d")
            config['tend'] = date.strftime("%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")
    
    # Initialize spectra object with config
    sp = spectra(config)
    
    # Get time intervals
    times = sp.get_time_intervals_static(
        UTCDateTime(config['tbeg']),
        UTCDateTime(config['tend'])+86400,  # add one day to the end time
        t_interval=86400,  # one day interval
        t_overlap=0,
    )
    
    # Process each time window
    for t1, t2 in times:
        print(f"Processing {t1.date}")
        
        for seed in config.get('seeds', None):

            # Create directory for figures (if it doesn't exist)
            net, sta, loc, cha = seed.split('.')
            filedir = f"{t1.year}/{cha}/spectra/"
            if not os.path.exists(f"{config.get('path_to_figures_out')}{filedir}"):
                os.makedirs(f"{config.get('path_to_figures_out')}{filedir}")

            try:
                # Load data stream
                st = sp.read_from_sds(
                    seed=seed,
                    path_to_archive=config['path_to_sds'],
                    tbeg=t1,
                    tend=t2+1,
                    merge=True,
                )
                
                if not st:
                    print(f"No data found for {t1.date}")
                    continue

                # Remove instrument response
                if config['remove_response']:
                    st = sp.remove_response(st, config['inventory_file'], output=config.get('output_type', None))

                # Add data to spectra object
                sp.add_trace(st[0])
                
                # Compute spectra in time windows as a collection
                sp.get_collection(
                    tinterval=config['tinterval'],
                    toverlap=config['toverlap'],
                    method=config['method'],
                )
                
                # apply averaging in fbands if true
                if config.get('apply_average'):
                    sp.get_fband_average(
                        fraction_of_octave=config.get('fraction_of_octave'),
                        average=config.get('averaging')
                    )
                
                # Classify collection quality
                if hasattr(config, 'quality_filter'):
                    sp.classify_collection_quality(threshold=config.get('threshold', 1e-15), # Threshold for high value detection (all above then considered bad)
                                              zero_seq_limit=config.get('zero_seq_limit', 20), # Maximum allowed length of zero sequences
                                              high_seq_limit=config.get('high_seq_limit', 20), # Maximum allowed length of high value sequences
                                              flat_seq_limit=config.get('flat_seq_limit', 20), # Maximum allowed length of flat-line segments
                )

                # Store spectra in day files
                sp.save_collection(config['path_to_data_out'])
                
                # Create output filename
                filename = f"{sp.tr_id.replace('.', '_')}_{str(t1.date).replace('-', '')}.png"
                filepath = os.path.join(config['path_to_figures_out'], filedir)
                
                # Plot and save
                sp.plot_spectra_and_helicorder(
                    fmin=config.get('fmin'),
                    fmax=config.get('fmax'),
                    cmap=config.get('cmap', 'rainbow'),
                    alpha=config.get('alpha', 0.8),
                    data_unit=config.get('data_unit', None), # Data unit
                    quality_filter=config.get('quality_filter', 'good'),
                    savefig=filepath + filename,
                    out=False,
                    show=False,
                )
                
                gc.collect()

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()




