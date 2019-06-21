from wmp.analyze import faces
import glob


def sample_lfw():
    LFW = "../data/lfw"
    SAMPLE = "../data/candidate-images/google-image-query/others"
    faces.sample_images(LFW, SAMPLE, 100)


def encode_single():
    PERSON = "../data/candidate-images/google-image-query/others"
    OUT_FP = "../data/candidate-images/google-image-query/others.json"
    encodings_dict = faces.encode_person(PERSON)
    faces.save_encodings(encodings_dict, OUT_FP)


def encode_all():
    ROOT_FACES = "../data/candidate-images/google-image-query"
    persons = f"{ROOT_FACES}/*/"

    for p in glob.glob(persons):
        name = faces.get_last_str(p)
        out_fp = f"{ROOT_FACES}/{name}.json"
        encodings_dict = faces.encode_person(p)
        faces.save_encodings(encodings_dict, out_fp)


def main():
    encode_single()


if __name__ == "__main__":
    main()
