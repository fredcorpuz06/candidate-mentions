from wmp.collect.webpage import GoogleImages
from wmp.analyze import faces
import glob
import os


def main():
    WEBDRIVER_FP = "../data/chromedriver_win32/chromedriver.exe"
    OUT_FOLDER = "../data/google-images"
    POLITICIANS_LIST = "../notes/politicians.txt"

    # QUERIES = [
    #     "Kamala Harris"]

    # Download images into own directory
    with open(POLITICIANS_LIST, "r") as f:
        queries = f.readlines()
        queries = [q.rstrip("\n") for q in queries]

    gi = GoogleImages(WEBDRIVER_FP, OUT_FOLDER)

    for q in queries:
        print(f"Google Query: {q}\n")
        gi.image_search(q, nimages=400)

    # Crop faces + put into a "cropped" subdirectory

    # OUT_FOLDER = "../data/google-images/Kamala_Harris/cropped"
    for query in glob.glob(f"{OUT_FOLDER}/*/"):
        cropped_folder_loc = f"{query}/cropped"
        gi._create_folder(cropped_folder_loc)
        for img in glob.glob(f"{query}/*.*"):
            try:
                faces.crop_face(img, cropped_folder_loc)
            except ValueError as e:
                os.remove(img)
                print(f"--- {img} deleted. Raised {e}")

    # Compare difference to a baseline image + delete mistakes


if __name__ == "__main__":
    main()
