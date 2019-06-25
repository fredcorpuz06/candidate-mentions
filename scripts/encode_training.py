from wmp.analyze import faces
import glob


def sample_lfw(n=250):
    LFW = "../data/lfw"
    SAMPLE = "../data/candidate-images/google-image-query/others"
    faces.sample_images(LFW, SAMPLE, n)


def encode_single(name):
    person = f"../data/candidate-images/google-image-query/{name}"
    out_fp = f"../data/candidate-images/google-image-query/{name}.json"
    encodings_dict = faces.encode_person(person)
    faces.save_encodings(encodings_dict, out_fp)
    print(f"***** Finished encoding {name} *****")


def encode_all():
    ROOT_FACES = "../data/candidate-images/google-image-query"
    persons = f"{ROOT_FACES}/*/"

    for p in glob.glob(persons):
        name = faces.get_last_str(p)
        out_fp = f"{ROOT_FACES}/{name}.json"
        encodings_dict = faces.encode_person(p)
        faces.save_encodings(encodings_dict, out_fp)
        print(f"***** Finished encoding {name} *****")


def main():
    #     encode_single("bclinton")
    #     encode_single("braun")
    #     encode_single("hclinton")
    #     encode_single("heller")
    #     encode_single("manchin")
    #     encode_single("mcconnell")
    #     encode_single("mueller")
    #     encode_single("obama")
    #     sample_lfw(n=250)
    encode_single("others")
#     encode_single("pelosi")
#     encode_single("ryan")
#     encode_single("sanders")
#     encode_single("trump")


if __name__ == "__main__":
    main()
