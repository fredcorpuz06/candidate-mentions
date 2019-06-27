def wmv_to_flac(vid_fp):
    '''Extracts audio from a WMV video file into a FLAC file of same name.'''
    path, file = os.path.split(vid_fp)
    audio_name = f"{file[:-4]}.flac"
    audio_fp = f"{path}/{audio_name}"

    ffmpeg_cmd = f"ffmpeg -y -i {vid_fp} -vn -f flac -ar 48000 -ab 128000 -ac 1 {audio_fp}"
    print(f"Converting to {audio_fp}")
    try:
        os.system(ffmpeg_cmd)  # ret is what is returns on the command line
    except Exception as e:
        print(f"{vid_fp} ==> {e}")

    return audio_name
