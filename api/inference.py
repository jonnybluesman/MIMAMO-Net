"""
Main custom class for using the MIMAMO-Net for inference.
"""

import os
import logging
from shutil import rmtree

import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable as Variable

from utils.pyutils import create_dir
from steerable.utils import get_device
from mimamo_net import Two_Stream_RNN
from video_processor import VideoProcessor
from resnet50_extractor import Resnet50Extractor
from sampler.snippet_sampler import Snippet_Sampler
from phase_difference_extractor import Phase_Difference_Extractor

logger = logging.getLogger('mimamo.inference')


class DeepFacialEmotionInference(object):

    def __init__(self,
         # parameters for testing
         model_path, batch_size, device=None, workers=0,
         # parameters for the video processing
         save_size=112, nomask=True, grey=False,
         quiet=True, tracked_vid=False, noface_save=False,
         openface_exe='OpenFace/build/bin/FeatureExtraction',
         # parameters for deep feature extraction (Resnet50)
         benchmark_dir='pytorch-benchmarks',
         model_name='resnet50_ferplus_dag',
         feature_layer='pool5_7x7_s1',
         # parameters for snipper sampler
         num_phase=12, phase_size=48,
         length=64, stride=64,
         # parameters for phase difference extractor
         height=4, nbands=2, scale_factor=2,
         extract_level=[1,2]):
        
        assert os.path.exists(model_path), \
            "Please, download the model checkpoint first."

        self.batch_size = batch_size
        self.workers = workers
        self.num_phase = num_phase
        self.phase_size = phase_size
        self.length = length
        self.stride = stride
        
        self.device = get_device(device)
        # Face detection and face alignment
        self.video_processor = VideoProcessor(
            save_size, nomask, grey, quiet,
            tracked_vid, noface_save, openface_exe)
        # From snippets to deep facial features
        self.resnet50_extractor =  Resnet50Extractor(
            benchmark_dir, self.device, model_name, feature_layer)
        # Phase and phase differences over time on faces
        self.pd_extractor = Phase_Difference_Extractor(
            height, nbands, scale_factor, 
            extract_level, self.device, not quiet)
        
        self.model = Two_Stream_RNN()  # model for FER
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.eval()
        self.model.to(self.device)
        
        logger.info(f"Loaded checkpoint from {model_path},"
                    f"Epoch:{checkpoint['epoch']}")

        self.label_name = ['valence', 'arousal'] # model output format


    def run_inference_from_video(self, input_video, keep_tmp=True):
        """
        Perform Video-Facial-Emotion recognition on the provided video.
        
        Args:
            input_video (str): path to the video stream to process.
        
        Returns:
            emotions_dict: a dict of dataframes containing the emotion 
                prediction of each video (key by name) per frame.

        Notes:
            - The user can provide a dir for temporary files (snippets, features).
        """
        video_name = os.path.splitext(os.path.basename(input_video))[0]
        tmp_dir = create_dir(os.path.join(
            os.path.dirname(input_video), video_name+"-tmp"))

        # first, the input video is processed using OpenFace
        opface_output_dir = os.path.join(tmp_dir, video_name+"_opface")
        self.video_processor.process(input_video, opface_output_dir)
        logger.info(f"{video_name} processed with OpenFace.")
        
        # the cropped and aligned faces are then fed to resnet50 for deep feature ext
        feature_dir = os.path.join(tmp_dir, video_name+"_pool5")
        self.resnet50_extractor.run(opface_output_dir, feature_dir, video_name=video_name)
        logger.info(f"Deep facial features extracted with pre-trained ResNet.")
        
        # creating a sequence of inputs for the NN (sampling images)
        dataset = Snippet_Sampler(
            video_name, opface_output_dir, feature_dir,
        	annot_dir = None, label_name = 'valence_arousal',
            test_mode = True, num_phase=self.num_phase,
            phase_size = self.phase_size, length=self.length, stride=self.stride)
        
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, 
            num_workers=self.workers, pin_memory=False)
        
        av_dict = self.run_inference_from_dataloader(data_loader, self.model)
        logger.info(f"{len(data_loader)} batches for {video_name}.")

        if not keep_tmp:  # tmp folders need to be removed
            logger.info(f"Removing {tmp_dir} as requested.")
            rmtree(tmp_dir)  # this removes nested folders too
        
        assert len(av_dict) == 1, "This function processes one video only."
        return list(av_dict.values())[0]  # the first and only item is returned


    def run_inference_from_dataloader(self, dataloader, train_mean=None, train_std=None):
        """
        Perform inference on a sequence of (already pre-processed) samples,
        provided as a torch dataloader to simplify processing.
        
        Args:
            dataloader (data.DataLoader): a dataloader containing video features.
            train_mean (list): mean per video, or None
            train_std (list): std per video, or None
        
        Returns:
            video_dict (dict): valence-arousal predictions per video (the name
                of each video is used as a key in the dictionary), each provided
                as a pandas DataFrame, and w.r.t. each frame.
        """
        sample_names = []
        sample_preds = []
        sample_ranges = []
        
        for i, data_batch in enumerate(dataloader):
            
            phase_f, rgb_f, label, ranges, names = data_batch
            with torch.no_grad():  # instantiating tensors for current batch
                phase_f = phase_f.type('torch.FloatTensor').to(self.device)
                phase_0, phase_1 = self.phase_diff_output(phase_f, self.pd_extractor)
                rgb_f = Variable(rgb_f.type('torch.FloatTensor').to(self.device))
                phase_0 = Variable(phase_0.type('torch.FloatTensor').to(self.device))
                phase_1 = Variable(phase_1.type('torch.FloatTensor').to(self.device))
            
            output = self.model([phase_0,phase_1], rgb_f)
            sample_names.append(names)
            sample_ranges.append(ranges)
            sample_preds.append(output.cpu().data.numpy())
        
        sample_names = np.concatenate([arr for arr in sample_names], axis=0)
        sample_preds = np.concatenate([arr for arr in sample_preds], axis=0)
        n_sample, n_length, n_labels = sample_preds.shape
        
        if train_mean is not None and train_std is not None:
            # standardise output features if required (mean and std provided)
            trans_sample_preds = sample_preds.reshape(-1, n_labels)
            trans_sample_preds = np.array(
                [correct(trans_sample_preds[:, i], train_mean[i], train_std[i]) 
                 for i  in range(n_labels)])  # scaling of predictions
            sample_preds = trans_sample_preds.reshape(n_sample, n_length, n_labels)
        
        sample_ranges = np.concatenate([arr for arr in sample_ranges], axis=0)
        video_dict = {}  # one entry per video, based on the dataloader provided
        
        for video in sample_names:
            mask = sample_names == video
            video_ranges = sample_ranges[mask]
            if video not in video_dict.keys():
                max_len = max([ranges[-1] for ranges in video_ranges])
                video_dict[video] = np.zeros((max_len, n_labels))
            video_preds = sample_preds[mask]
            
            min_f, max_f = 0, 0  # make sure to return full range of video frames
            for rg, pred in zip(video_ranges, video_preds):
                start, end = rg
                video_dict[video][start:end, :] = pred
                min_f = min(min_f, start)
                max_f = max(max_f, end)
            assert (min_f == 0) and (max_f == max_len)
        
        for video in video_dict.keys():  # creating a dataframe per video
            video_dict[video] = pd.DataFrame(
                data=video_dict[video], columns=self.label_name)
        
        return video_dict


    def phase_diff_output(self, phase_batch, steerable_pyramid):
        """
        Extract the first level and the second level phase difference images.
        """
        sp = steerable_pyramid
        bs, num_frames, num_phases, W, H = phase_batch.size()
        
        coeff_batch = sp.build_pyramid(phase_batch.view(bs*num_frames, num_phases, W, H))
        assert isinstance(coeff_batch, list)
        
        phase_batch_0 = sp.extract(coeff_batch[0])
        N, n_ch, n_ph, W, H= phase_batch_0.size()
        
        phase_batch_0 = phase_batch_0.view(N, -1, W, H)
        phase_batch_0 = phase_batch_0.view(bs, num_frames, -1, W, H)
        phase_batch_1 = sp.extract(coeff_batch[1])
        
        N, n_ch, n_ph, W, H = phase_batch_1.size()
        phase_batch_1 = phase_batch_1.view(N, -1, W, H)
        phase_batch_1 = phase_batch_1.view(bs, num_frames, -1, W, H)
        
        return phase_batch_0, phase_batch_1
