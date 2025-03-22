import os
from tqdm import tqdm
import yaml
from typing import Callable, Iterable, Sequence, Tuple
import pathlib

def slakh(audio_path, midi_path, extensions):
    tracks = [os.path.join(audio_path, subfolder) for subfolder in os.listdir(audio_path)]
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



def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm


def search_for_audios(
    path_list: Sequence[str],
    extensions: Sequence[str] = [
        "wav", "opus", "mp3", "aac", "flac", "aif", "ogg"
    ],
):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f"*.{ext}"))
    audios = flatten(audios)
    audios = [str(a) for a in audios if 'MACOS' not in str(a)]
    return audios



def simple_audio(audio_folder, midi_folder, extensions, exclude):
    audio_files = search_for_audios([audio_folder], extensions = extensions)
    audio_files = map(str, audio_files)
    audio_files = map(os.path.abspath, audio_files)
    audio_files = [*audio_files]

    audio_files = [
        f for f in audio_files
        if not any([excl in f for excl in exclude])
    ]
    metadatas = [{"path": audio} for audio in audio_files]
    midi_files = [None] * len(audio_files)
    print(len(audio_files), " files found")
    return audio_files, midi_files, metadatas
    

def simple_midi(audio_folder, midi_folder, extensions, exclude):
    audio_files, _, _ = simple_audio(audio_folder, midi_folder, extensions, exclude)
    
    midi_func = lambda x: x.replace(audio_folder, midi_folder).replace("wav", "mid")
    midi_files = map(midi_func, audio_files)
    midi_files = [*midi_files]
    
    metadatas = [{"path": audio, "midi_path": midi} for audio, midi in zip(audio_files, midi_files)]
    
    return audio_files, midi_files, metadatas



def get_parser(parser_name):
    if parser_name == "simple_audio":
        return simple_audio
    elif parser_name == "simple_midi":
        return simple_midi
    if parser_name == "slakh":
        return slakh
    else:
        raise ValueError(f"Parser {parser_name} not available")
