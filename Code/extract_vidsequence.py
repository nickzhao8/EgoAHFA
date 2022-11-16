import pandas as pd
import ffmpeg
import os
from pathlib import Path

def vid_process (infile, outfile, start=0, end=0):
	"""Uses ffmpeg to filter and trim a video. Filters: fps (30), spp (compress), scale (360p)
	Arguments:
		infile (string): path to the input video file (including file extension)
		outfile (string): path to the output video file (including file extension)
		start (int): start of trim in seconds
		end (int): end of trim in seconds
	"""
	(
	ffmpeg
	.input(infile)
	.filter('fps', fps=32)
	.filter('spp')					# Compress
	.filter('scale', w=852, h=480)	# Scale to 480p
	.trim(start=start, end=end)
	.setpts('PTS-STARTPTS')
	.output(outfile, **{'qscale:v': 3})
	# .output(outfile)
	.run(quiet=True, overwrite_output=True)
	)

# Define file paths
datapath = Path("M:/Wearable_Hand_Monitoring/CODE AND DOCUMENTATION/Nick Z/HomeLab_GRASSP_Annotated.xlsx")
video_repo = Path(r'M:\Wearable_Hand_Monitoring Datasets_DONOTMODIFY\DATASET_SCI')
out_root = Path(r'C:\Users\zhaon\Documents\GRASSP_JPG_FRAMES')

# Read Excel spreadsheet
xlFile = pd.read_excel(datapath, sheet_name=None, engine='openpyxl')

# Create output parent directory
os.makedirs(out_root, exist_ok=True)


# Iterate over each sheet (subject)
for sheet in xlFile.keys():
	# Skip non-subject sheets
	if "Sub" not in sheet: continue
	
	sub_num = sheet.split("Sub")[1]
	# FIXME: DEBUG: - start at later subject
	# if int(sub_num) < 7: continue
	# if int(sub_num) > 8: break

	print("=== Participant: ", sheet, ' ===')

	# Create parent subject directory
	os.makedirs(f'{out_root}/{sheet}', exist_ok=True)
	# Create directories for each class (scores 0-5)
	# for i in range(6):
		# os.makedirs(f'{out_root}/{sheet}/{str(i)}', exist_ok=True)

	start_times = xlFile[sheet]['Start Time']
	end_times = xlFile[sheet]['End Time']
	scores = xlFile[sheet]['GRASSP Score']
	videos = xlFile[sheet]['Video']
	tasks = xlFile[sheet]['Task']
	start_seconds = []
	end_seconds = []
	# Convert start/end times to seconds
	for i in range(len(start_times)):
		start_seconds.append(int(start_times[i].split(":")[0])*60 + int(start_times[i].split(":")[1]))
		end_seconds.append(int(end_times[i].split(":")[0])*60 + int(end_times[i].split(":")[1]))
	
	sub_dir = str(sub_num).zfill(2)

	# Task counter
	task_counter = {}
	
	vid_ind = 0
	# Iterate over each video
	for vid in list(dict.fromkeys(videos)): # Only contains unique elements from videos
		if isinstance(vid, int): vid_name = str(vid).zfill(2) + ".MP4"
		else: vid_name = str(vid) + ".MP4"
		vid_path = Path(video_repo, sub_dir, "GoPro", "Centre", vid_name)

		print("Source video: ", vid_name)

		# Process the video into compressed video segments, separated into ADLs
		while (vid_ind < len(videos) and videos[vid_ind] == vid):
			out_path = Path(out_root, sheet, str(scores[vid_ind]))
			if (not os.path.exists(out_path)): os.makedirs(out_path) # Create score directory only if necessary
			if tasks[vid_ind] not in task_counter: task_counter[tasks[vid_ind]] = 0
			else: task_counter[tasks[vid_ind]] += 1
			outviddir = Path(out_path, f"sub{sub_num}_{tasks[vid_ind]}_{task_counter[tasks[vid_ind]]}".replace(' ','_'))
			os.makedirs(outviddir, exist_ok=True) # Remove for vid output
			print(f"Processing: {tasks[vid_ind]} {task_counter[tasks[vid_ind]]}")
			vid_process(
				infile 	= str(vid_path), 
				outfile = str(Path(outviddir, "%04d.jpg")), 
				# outfile = str(outviddir) + '.mp4', 
				start 	= start_seconds[vid_ind], 
				end		= end_seconds[vid_ind]
			)
			vid_ind += 1
	


