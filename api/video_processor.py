import os
import logging


logger = logging.getLogger('mimamo.video_processor')


class VideoProcessor(object):

    def __init__(self, size=112, nomask=True, grey=False, quiet=True,
                 tracked_vid=False, noface_save=False,
                 openface_exe = 'OpenFace/build/bin/FeatureExtraction'):
        """ 
        Video Processor using OpenFace for face detection and alignment.
        Given an input video, this processor will create a directory where all
        cropped and aligned faces are saved.

        Parameters: 
            size: int, default 112
                The output faces will be saved as images where the width and 
                height are size pixels.
            nomask: bool, default True
                If True, the output face image will not be masked (mask the 
                region except the face). Otherwise, the output face image will 
                be masked, not containing background.
            grey: bool, default False
                If True, the output face image will be saved as greyscale images
                instead of RGB images.
            quiet: bool, default False
                If False, will print out the processing steps live.
            tracked_vid: bool, default False
                If True, will save the tracked video, which is an output video
                with detected landmarks.
            noface_save: bool, default False
                If True, those frames where face detection is failed will be 
                saved (blank image); 
                else those failed frames will not saved.
            OpenFace_exe: String, default uses 'FeatureExtraction'
                By default, the OpenFace library is installed in the same 
                directory as Video_Processor. It can be changed to the current
                OpenFace executable file.

        Notes:
            - The openface executable to consider should the first argument,
                which should mandatory as it depends on the project conf.
            - If more than a single person in present in the video, using the
                normal execution mode of openface will either: (i) select one
                person and ignore the others, in case they appear in the same
                frame(s); (ii) concatenate the faces of multiple persons in a 
                a frame by frame manner, in case they do not occur in the same
                frame(s). To solve this, the "Multi" execution mode should be
                considered, although some post-processing would be needed (check
                additional openface arguments that might probably do this already)
                to ensure that each person is associated to a separate folder.
            - The video processor should also return a list of frames ids where
                a face was detected, following the conventions of openface. In
                this manner, it will be possible to infer the time of each frame.

        """
        self.size = size
        self.nomask = nomask
        self.grey = grey
        self.quiet = quiet
        self.tracked_vid = tracked_vid
        self.noface_save = noface_save
        self.openface_exe = openface_exe

        if not isinstance(self.openface_exe, str) or not os.path.exists(self.openface_exe):
            raise ValueError("Openface has to be string object and needs to exist.")
        self.openface_exe = os.path.abspath(self.openface_exe)


    def process(self, input_video, output_dir=None):
        '''        
        Arguments:
            input_video: String
               The input video path, or the input sequence directory, where each
               image representing a frame, e.g. 001.jpg, 002.jpg, ... 200.jpg
            output_dir: String, default None
               The output faces will be saved in output_dir. By default the 
               output_dir will be in the same parent directory of the video.
        '''

        if not isinstance(input_video, str) or not os.path.exists(input_video):
            raise ValueError("Input video has to be string object and needs to exist.")
        if os.path.isdir(input_video):
            assert len(os.listdir(input_video)) > 0, \
                "Input sequence directory {} cannot be empty".format(input_video)
            arg_input = '-fdir'
        else:
            arg_input = '-f'

        if not isinstance(output_dir, str) and output_dir:
            raise ValueError("Output directory either None or string")

        input_video = os.path.abspath(input_video)

        if output_dir is None:  # the name of the output dir is inferred
            output_dir = os.path.join(os.path.dirname(input_video), 
                os.path.basename(input_video).split('.')[0])

        if not os.path.exists(output_dir):  # creating output dir
            os.makedirs(output_dir)
        else:  # the output directory already exists, no extraction needed
            logger.info(f"Output dir: {output_dir} already exists."
                "No video processing will be performed.")
            return


        opface_options = " {} ".format(arg_input) + input_video + " -out_dir " \
            + output_dir + " -simsize " + str(self.size)
        opface_options += " -2Dfp -3Dfp -pdmparams -pose -aus -gaze -simalign "

        if not self.noface_save:
            opface_options +=" -nobadaligned "
        if self.tracked_vid:
            opface_options +=" -tracked "
        if self.nomask:
            opface_options+= " -nomask"
        if self.grey:
            opface_options += " -g"
        if self.quiet:
            opface_options += " -q"
        

        call = self.openface_exe + opface_options
        os.system(call)
