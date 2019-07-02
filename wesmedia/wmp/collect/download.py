'''Download files from link.
'''

import re
import requests
import urllib


class FileDownload:

    def __init__(self):
        pass

    def download_image(self, uri, folder, file_name, default_format=None):
        '''Downloads image to specified file name, preserving file type.'''
        if not uri:
            return None

        file_ext = self.find_file_format(uri, default_format=default_format)
        file_path = f"{folder}/{self._clean_name(file_name)}.{file_ext}"
        try:
            img_data = requests.get(uri).content
        except (ConnectionError, Exception) as e:
            print(f"--- {uri} caused error {e}")
            return None
        else:
            with open(file_path, "wb+") as f:
                f.write(img_data)
            print(f"Downloaded {file_path}")
            return f"{file_name}.{file_ext}"

    def download_video(self, uri, folder, file_name):
        '''Downloads video to specified file name, preserving file type.'''
        if not uri:
            return None

        file_ext = self.find_file_format(uri)
        file_path = f"{folder}/{self._clean_name(file_name)}.{file_ext}"
        try:
            urllib.request.urlretrieve(uri, file_path)
            print(f"Downloaded {file_path}")
        except Exception as e:
            print(f"--- {uri} caused error {e}")
            return None
        else:
            return f"{file_name}.{file_ext}"

    def find_file_format(self, uri, default_format=None):
        '''Extract file format of file to be downloaded.'''
        # uri = "https://scontent.fbed1-1.fna.fbcdn.net/v/t39.16868-6/55301243_2079741458747847_5772627081674358784_n.jpg?_nc_cat=101&_nc_oc=AQltQByMk6wIE6OJxjMpxaXVAUJnytEVJaZepHOboxtZonp6k0iodF9fA2QiYYY84XI&_nc_ht=scontent.fbed1-1.fna&oh=6bf632849e2ac53cf9edddba5fbf14ee&oe=5DC11F7E"
        # uri = "https://video.xx.fbcdn.net/v/t42.9040-2/43676685_2169699926374656_1057658348297519104_n.mp4?_nc_cat=101&_nc_oc=AQkoV3zJHJRZNyAl3Ur6FopMNzxGHvUnHMhYiijlEp30FMWDNm4ckdydHfQ3_bVOSVA&_nc_ht=video.fbed1-1.fna&oh=0cc6bfd5e88e304377552e654f8aa52f&oe=5D1537CD"
        try:
            file_format = re.findall(r"(?<=\.)[a-z0-9]+(?=\?|$)", uri)[0]
        except IndexError:
            if default_format:
                file_format = default_format
            else:
                file_format = ""
        finally:
            return file_format

    def _clean_name(self, file_path):
        '''Ensure valid file name'''
        return re.sub("[&?=,']", "_", file_path)
