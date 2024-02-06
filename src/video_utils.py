import subprocess
import numpy as np 
import os
from gpmfstream import Stream
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
from reconstruction_utils import get_legend

def get_video_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def extract_frames_and_gopro_gravity_vector(video_names, timestamps, width, height, fps, tmp_dir):
    """As input, takes a list of video names of the form: [<video_name1>, <video_name2>], 
    and a list of (minute:second)-timestamps of the example form ["<seconds>-end","begin-<seconds>"]
    using FFMPEG, extracts frames at <fps> frames per second, with the height and width set accordingly,
    and in reverse for videos where the camera moves backwards.
    """

    os.makedirs(tmp_dir + "/rgb", exist_ok=True)
    
    gravity_vectors = []
    total_frames = 0

    for video_id, (video_name, timestamp) in enumerate(zip(video_names, timestamps)):
        
        targetpath = tmp_dir + "/" + video_name.split("/")[-1].split(".")[0].replace(" ", "_").replace("/", "_")
        os.makedirs(targetpath, exist_ok=True)
        
        begin, end = timestamp.split("-")
        ss = ""
        to = ""
        if begin != "begin":
            ss += " -ss " + begin
        if end != "end":
            to += " -to " + end
        
        # First: cut video
        os.system("ffmpeg -hide_banner -loglevel error"+ss+" " +to+" -y -i '"+video_name+"'  -c copy "+tmp_dir+"/"+str(video_id)+".mp4")
        # Second: scale video to right dimensions
        os.system("ffmpeg -hide_banner -loglevel error  -y -i "+tmp_dir+"/"+str(video_id)+".mp4 -vf scale="+str(width)+":"+str(height)+" "+tmp_dir+"/"+str(video_id)+"_.mp4")
        # Third: extract frames
        os.system("ffmpeg -hide_banner -loglevel error -y -i "+tmp_dir+"/"+str(video_id)+"_.mp4 -vf fps="+str(fps)+" -qscale:v 2 "+targetpath +"/%07d.jpg")
        num_frames = len(os.listdir(targetpath))
        
        for frame in os.listdir(targetpath):
            frameid = int(frame.split(".")[0])
            os.system("mv "+targetpath + "/" + frame + " "+tmp_dir+"/rgb/" + str(frameid + total_frames).zfill(7) + ".jpg")

        total_frames += num_frames
        gravity_vectors.append(get_gravity_vectors(video_name, timestamp, num_frames))
    if gravity_vectors[0] is None:
        return None
    return np.concatenate(gravity_vectors)


def get_gravity_vectors(video, timestamp, number_of_frames):
    """Uses gpmfstream to extract gravity vectors from an MP4 video file."""
    try:
        grav = Stream.extract_streams(video)["GRAV"].data
    except Exception as e:
        print("WARNING: Could not extract gravity vectors from video file:", video,  " is your video an unedited GoPro video?")
        return None    
    length = get_video_length(video)

    begin, end = timestamp.split("-")
    #TODO: timestamp parsing!
    if begin == "begin":
        begin = 0
    if end == "end":
        end = length
    begin = float(begin)
    end = float(end)

    grav = grav[int(begin/length*len(grav)):int(end/length*len(grav))]
    inds = np.linspace(0, len(grav)-1,  number_of_frames).astype(np.int32)
    grav = grav[inds]
    grav /= np.linalg.norm(grav, axis=1).reshape(-1, 1)
    return grav



def render_video(img_list, depths, semantic_segmentation, results_npy, fps, class_to_label, label_to_color, tmp_dir):
    """Renders a video from the given images, depths, semantic_segmentation and 2d maps."""
    os.makedirs(tmp_dir + "/render", exist_ok=True)
    
    # For visualization, its nicer when depths are scaled between 0 and 1, and sqrt_scaled
    q95 = np.quantile(depths.reshape(-1), 0.95)
    q5 = np.quantile(depths.reshape(-1), 0.05)
    depths = np.clip(depths, q5, q95)
    depths_ = np.sqrt(depths)
    depths_ /= np.max(depths_)

    color_semseg = np.zeros((semantic_segmentation.shape[0], semantic_segmentation.shape[1], semantic_segmentation.shape[2], 3), dtype=np.uint8)
    
    
    for class_name, class_label in class_to_label.items():
        color_semseg[semantic_segmentation==class_label] = label_to_color[class_label]

    class_to_color = {class_name: label_to_color[class_label] for class_name, class_label in class_to_label.items()}
    legend = get_legend(class_to_color, tmp_dir)

    final_rgb = results_npy[:,:,1:4]
    final_class_rgb = results_npy[:,:,6:9]
    frame_index = results_npy[:,:,-3:-2].astype(np.int16)
    for i in tqdm(range(len(depths))):
        
        rgb = np.array(Image.open(img_list[i]))/255.
        
        ind = (frame_index <= i).astype(np.uint8)
        results_npy_rgb = final_rgb * ind
        results_npy_class_rgb = final_class_rgb * ind


        if results_npy_rgb.shape[0]<results_npy_rgb.shape[1]:
            results_npy_rgb = np.concatenate([results_npy_rgb,results_npy_class_rgb],axis=0)
        else:
            results_npy_rgb = np.concatenate([results_npy_rgb,results_npy_class_rgb],axis=1)

        if i == 0:
            ratio = legend.shape[0] / results_npy_rgb.shape[0]
            legend = resize(legend, (round(legend.shape[0]/ratio), round(legend.shape[1]/ratio)))
        results_npy_rgb = np.concatenate([legend, results_npy_rgb/255.], axis=1).transpose(1, 0, 2)
        results_npy_rgb = resize(results_npy_rgb, rgb.shape[:2])
        
        resize_ratio = results_npy_rgb.shape[1]/rgb.shape[1]
        results_npy_rgb = resize(results_npy_rgb, (round(results_npy_rgb.shape[0]/resize_ratio), round(results_npy_rgb.shape[1]/resize_ratio)))

        image = np.concatenate([
            np.concatenate([rgb, 0.2 * rgb + 0.8 * plt.cm.seismic(depths_[i])[:,:,:3]], axis=0),
            np.concatenate([0.3 * rgb + 0.7 * color_semseg[i].astype(np.float32)/255., 
                            results_npy_rgb]
                           , axis=0),
        ], axis=1)
        plt.imsave(tmp_dir + "/render/" + str(i).zfill(7)+".jpg", image)
    os.system("ffmpeg  -hide_banner -loglevel error -framerate "+str(fps)+" -pattern_type glob -i '"+tmp_dir+"/render/*.jpg' \
          -c:v libx264 -pix_fmt yuv420p "+tmp_dir+"/out.mp4")
