
import os
import argparse
from time import sleep

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from inference import DeepFacialEmotionInference
from utils.pyutils import create_dir, is_dir, is_file, \
    set_logger, get_files_with_extension


def open_video(input_video):

    video_capture = cv2.VideoCapture(input_video)
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"No. of frames: {length} | FPS: {round(fps)}"
           "| Resolution: {width}x{height} detected.")

    return video_capture, fps, (length, width, height)


def plot_av_predictions(results_df, save_dir=None, name=''):

    fig, ax = plt.subplots(figsize=(20, 5))
    sns.set_style("darkgrid")

    sns.lineplot(x="s", y="arousal", data=results_df, label="arousal", ax=ax)
    sns.lineplot(x="s", y="valence", data=results_df, label="valence", ax=ax)

    ax.set_ylabel("arousal | valence (standardised)")
    ax.set_title("Arousal and valence variation in time")

    if save_dir:  # the plot will be saved in the given folder
        name = name + "_" if len(name) > 0 else name
        fig.savefig(os.path.join(save_dir, f"{name}va_trend.png"), dpi=300)

    return fig, ax


def process_video(input_video, engine, out_dir, smooth_window=.5, keep_tmp=True):

    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    video_capture, fps, _ = open_video(input_video)
    video_name = os.path.splitext(os.path.basename(input_video))[0]

    results_df = engine.run_inference_from_video(input_video, keep_tmp)
    results_df = results_df.apply(moving_average, axis=0, 
        w=int(smooth_window*round(fps))) if smooth_window else results_df

    results_df['s'] = (results_df.index / round(fps))
    results_df.to_csv(os.path.join(out_dir, 
        f"{video_name}_va_preds.csv"), index=None)
    plot_av_predictions(results_df, out_dir, video_name)

    return results_df


def main():
    
    parser = argparse.ArgumentParser(
        description="Facial Emotion Recognition from videos, using MIMAMO-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    file_tp, dir_tp = lambda x: is_file(parser, x), lambda x: is_dir(parser, x)

    parser.add_argument("video_path", action="store", type=str,
                        help="Path to a video file to process for FER.")
    parser.add_argument("--out_dir", action="store", type=str,
                        help="A directory where the results will"
                             " be saved (will be created in setup_dir if None).")
    parser.add_argument("--smoothing_size", action="store", type=float,
                        help="Size (in seconds) of the moving-average smoothing.")

    # Arguments for the inference step with MIMAMO-Net
    parser.add_argument("--model_path", action="store", type=file_tp,
                        help="Checkpoint of the trained MIMAMO network to use.")
    parser.add_argument("--batch_size", action="store", type=int, default=1,
                        help="The number of face snippets to process at once.")
    parser.add_argument("--n_workers", action="store", type=int, default=1,
                        help="The number of threads to use for data loading.")
    parser.add_argument("--device", action="store", type=str, default="cpu",
                        help="The device to use for processing and inference.")

    # Arguments for face detection-recognition-aligment with OpenFace
    parser.add_argument("--openface_path", action="store", type=dir_tp,
                        help="Path to the OpenFace binaries for feature ext.")
    parser.add_argument("--multi_face", action="store_true",
                        help="Whether to enable multi-face recognition.")
    parser.add_argument("--face_size", action="store", type=int, default=112,
                        help="Size to consider for each face detected.")
    parser.add_argument("--mask", action="store_true",
                        help="Whether the output face image will be masked"
                        " (mask the region except the face). Otherwise, the"
                        " output face image will not contain any background.")
    # Arguments for deep facial feature extraction (Resnet50)
    parser.add_argument("--benchmark_path", action="store", type=dir_tp,
                        help="Path to the pytorch-benchmarks folder.")
    parser.add_argument("--model_name", action="store", type=str,
                        default="resnet50_ferplus_dag", help="ResNet model.")
    parser.add_argument("--feature_layer", action="store", type=str,
                        default="pool5_7x7_s1", help="Feature layer name.")
    # Arguments for the snippet sampler
    parser.add_argument("--num_phase", action="store", type=int, default=12,
                        help="Number of phase difference images to consider.")
    parser.add_argument("--phase_size", action="store", type=int, default=48,
                        help="Phase image size, default is 48x48.")
    parser.add_argument("--length", action="store", type=int, default=64,
                        help="The length of snippets returned.")
    parser.add_argument("--stride", action="store", type=int, default=64,
                        help="The stride taken when sampling sequence of"
                             " snippets. If stride<length, the adjacent"
                             " sequence will overlap with each other.")
    # Arguments for the phase difference extractor
    parser.add_argument("--height", action="store", type=int, default=4,
                        help="The coefficients levels inc. low- and high pass")
    parser.add_argument("--nbands", action="store", type=int, default=2,
                        help="The number of orientations of bandpass filters.")
    parser.add_argument("--scale_factor", action="store", type=int, default=2,
                        help="Spatial resolution reduction scale.")

    # Simple utilities and running hooks
    parser.add_argument('--log', action='store_true',
                        help='Enables logging and printing on the console.')
    parser.add_argument("--logdir", action="store", type=str,
                        help="Path to a directory where logs will be saved.")
    parser.add_argument("--keep_tmp", action="store_true",
                        help="Whether to keep all temporary files after a new" 
                             " video has been successfully processed.")
    parser.add_argument('--watchdir', action='store_true',
                        help="Whether to run the script in watch mode.")
    parser.add_argument('--watch_interval', action='store', type=int, default=60,
                        help="Time interval (in seconds) betwwen checks.")
    parser.add_argument('--video_exts', action='store', nargs="+", default=[".mp4"],
                        help="Extension(s) of the video files to process.")
    
    args = parser.parse_args()
    set_logger("mimamo", args.log, log_dir=args.logdir)

    if args.out_dir is None:  # naming results dir in setup
        base_dir = os.path.dirname(args.video_path)
        args.out_dir = os.path.join(base_dir, "av-output")
    create_dir(args.out_dir)  # creating if does not exist

    assert args.model_path and args.openface_path and args.benchmark_path, \
        "Provide the paths to: the MIMAMO checkpoint file (--model_path)," \
        " the OpenFace binaries (--openface_path) and the Resnet50 benchmark" \
        " project for feature extraction (--benchmark_path)."
    
    args.openface_path = args.openface_path + "FeatureExtraction" \
        if not args.multi_face else args.openface_path + "FaceLandmarkVidMulti"
    
    inference_engine = DeepFacialEmotionInference(
        args.model_path, batch_size=args.batch_size, 
        device=args.device, workers=args.n_workers,
        # parameters for face processing with OpenFace
        openface_exe=args.openface_path,
        save_size=args.face_size, nomask=not(args.mask), 
        grey=False, quiet=True, tracked_vid=False, noface_save=False,
        # parameters for deep feature extraction (Resnet50)
        benchmark_dir=args.benchmark_path, model_name=args.model_name,
        feature_layer=args.feature_layer,
        # parameters for snipper sampler
        num_phase=args.num_phase, phase_size=args.phase_size,
        length=args.length, stride=args.stride,
        # parameters for phase difference extractor
        height=args.height, nbands=args.nbands, 
        scale_factor=args.scale_factor, extract_level=[1,2])

    while True:  # a deamon-like process that is trivially exited if not needed

        videos = get_files_with_extension(args.video_path, args.video_exts) \
            if os.path.isdir(args.video_path) else [args.video_path]
        outputs = [os.path.splitext(os.path.basename(csv_file))[0] \
            for csv_file in get_files_with_extension(args.out_dir, ['.csv'])]

        videos_to_process = [v for v in videos if os.path.splitext(
            os.path.basename(v))[0] + "_va_preds" not in outputs]
        print(f"Processing {len(videos_to_process)} out of {len(videos)} found")

        for video_path in videos_to_process:  # processing each video
            process_video(str(video_path), inference_engine, args.out_dir,
                smooth_window=args.smoothing_size, keep_tmp=args.keep_tmp)

        if not args.watchdir: break
        sleep(args.watch_interval)


if __name__ == "__main__":
    # execute only if run as a script
    main()