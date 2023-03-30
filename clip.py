from audioclipextractor import AudioClipExtractor, SpecsParser
import argparse
import sys
import os



def parse_args():
    desc = "Tool to create multiclass json labels file for stylegan2-ada-pytorch"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--verbose', action='store_true',
                        help='Print progress to console.')

    parser.add_argument('--song_path', type=str,
                        default='./',
                        help='Directory path to the inputs folder. (default: %(default)s)')

    parser.add_argument('--output_dir', type=str,
                        default='./',
                        help='Directory path to the outputs folder. (default: %(default)s)')
    parser.add_argument('--start', type=str,
                        default='0',
                        help='Directory path to the outputs folder. (default: %(default)s)')

    parser.add_argument('--end', type=str,
                        default='30',
                        help='Directory path to the outputs folder. (default: %(default)s)')
    parser.add_argument('--zip', type=str,
                        default=False,
                        help='Directory path to the outputs folder. (default: %(default)s)')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    song_path = args.song_path
    outdir = args.output_dir
    zip = bool(args.zip)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

        # Inicialize the extractor

    for filename in os.listdir(song_path):
        f = os.path.join(song_path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            ext = AudioClipExtractor(f)
            specs = f'{args.start}  {args.end} {filename[:-4]}'
            
    # Define the clips to extract
    # It's possible to pass a file instead of a string
          

            # Extract the clips according to the specs and save them as a zip archive
            ext.extract_clips(specs, outdir, zip_output=False, text_as_title=True )

if __name__ == "__main__":
    main()