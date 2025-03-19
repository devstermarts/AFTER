import os
from tqdm import tqdm
import yaml


def get_tracks_slakh(path):
    tracks = [os.path.join(path, subfolder) for subfolder in os.listdir(path)]
    meta = tracks[0] + "/metadata.yaml"
    ban_list = [
        "Chromatic Percussion",
        "Drums",
        "Percussive",
        "Sound Effects",
        "Sound effects",
    ]

    instr = []
    stem_list = []
    metadata = []
    total_stems = 0
    for trackfolder in tqdm(tracks):
        try:
            meta = trackfolder + "/metadata.yaml"
            with open(meta, "r") as file:
                d = yaml.safe_load(file)
            for k, stem in d["stems"].items():
                if stem["inst_class"] not in ban_list:
                    stem_list.append(trackfolder + "/stems/" + k + ".flac")
                    instr.append(stem["inst_class"])
                    metadata.append(stem)
                total_stems += 1
        except:
            print("ignoring reading folder : ", trackfolder)
            continue

    print(set(instr), "instruments remaining")
    print(total_stems, "stems in total")
    print(len(stem_list), "stems retained")

    audios = stem_list
    metadatas = [{
        "path": audio,
        "instrument": inst
    } for audio, inst in zip(audios, instr)]

    def get_midi_from_path(audio_path):
        split = audio_path.split("/")
        split[-2] = "MIDI"
        midi_path = "/".join(split)[:-5] + ".mid"
        return midi_path

    midis = [get_midi_from_path(audio) for audio in audios]
    return audios, midis, metadatas


def get_parser(parser_name):
    if parser_name == "slakh":
        return get_tracks_slakh
    else:
        raise ValueError(f"Parser {parser_name} not available")
