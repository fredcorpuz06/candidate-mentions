import json
import glob
import os
from wmp.analyze import faces
from wmp.analyze.faces import FacesError


def get_last_str(folder_path):
    strs = re.split("[^A-Za-z-]", folder_path)
    strs = list(filter(lambda x: x != "", strs))
    return strs[-1]


def main():
    names = []
    encodings = []
    persons = glob.glob("../data/candidate-images/google-image-query/*/")
    for p in persons:
        name = get_last_str(p)
        imgs_fps = f"{p}/*.jpg"
        for img_fp in glob.glob(imgs_fps):
            try:
                encoding = faces.encode_face(img_fp)
            except (ValueError, FacesError) as e:
                print(f"{img_fp} raised {e}")
            else:
                encodings.append(encoding)
                names.append(name)

    assert len(names) == len(encodings), "Names does not match encodings"

    people_encodings = {
        "name": names,
        "encoding": encodings
    }

    out_fp = "../output/poi_encodings.json"
    with open(out_fp, "w") as fp:
        json.dump(people_encodings, fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
