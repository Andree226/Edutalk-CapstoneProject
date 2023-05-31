import moviepy.editor as mp
cvt_video = mp.VideoFileClip("WIN_20230531_01_46_47_Pro.mp4")
ext_audio = cvt_video.audio
ext_audio.write_audiofile("audio_extracted.wav")
