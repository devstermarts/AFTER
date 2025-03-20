from typing import Callable, Iterable, Sequence, Tuple
import pathlib
import librosa
import lmdb
import torch
import numpy as np
from after.dataset.audio_example import AudioExample
from after.dataset.parsers import get_parser
import os
from tqdm import tqdm
from after.dataset.transforms import BasicPitchPytorch
import pickle
import pretty_midi
from absl import app, flags
import copy

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path',
    None,
    help=
    'Path to a directory containing audio files - use slakh main directory to use slakh',
    required=True)
flags.DEFINE_string('output_path',
                    ".",
                    help='Output directory for the dataset',
                    required=False)

flags.DEFINE_string(
    'parser',
    None,
    help=
    'File parser defined in parsers.py. Use None for recursive search of audio files in the input path',
    required=False)

flags.DEFINE_bool(
    'basic_pitch_midi',
    False,
    help='Use basic pitch to obtain midi scores from the audio files',
    required=False)

flags.DEFINE_multi_string('exclude', [],
                          help='kewords to exclude from the search',
                          required=False)

flags.DEFINE_bool('normalize',
                  True,
                  help='Normalize audio files magnitude',
                  required=False)

flags.DEFINE_bool('cut_silences',
                  True,
                  help='Remove silence chunks',
                  required=False)

flags.DEFINE_integer('num_signal',
                     262144,
                     help='Number of audio samples to use during training')

flags.DEFINE_integer('sample_rate',
                     44100,
                     help='Sampling rate to use during training')

flags.DEFINE_integer('db_size',
                     200,
                     help='Maximum size (in GB) of the dataset')

flags.DEFINE_string(
    'emb_model_path',
    None,
    help='Embedding model path for precomputing the AE embeddings',
    required=False)

flags.DEFINE_integer('batch_size', 8, help='Number of chunks', required=False)
flags.DEFINE_string('device',
                    "cpu",
                    help='device for emb_model',
                    required=False)

flags.DEFINE_multi_string(
    'ext',
    default=['wav', 'opus', 'mp3', 'aac', 'flac'],
    help='Extension to search for in the input directory')

flags.DEFINE_bool('dyndb',
                  default=False,
                  help="Allow the database to grow dynamically")

flags.DEFINE_bool('save_waveform',
                  default=False,
                  help="Save the waveform in the database")


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


def normalize_signal(x: np.ndarray,
                     max_gain_db: int = 30,
                     gain_margin: float = 0.9):
    peak = np.max(abs(x))
    if peak == 0:
        return x
    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)
    return gain_margin * x * gain


def get_midi(midi_data, chunk_number):
    length = FLAGS.num_signal / FLAGS.sample_rate
    tstart = chunk_number * FLAGS.num_signal / FLAGS.sample_rate
    tend = (chunk_number + 1) * FLAGS.num_signal / FLAGS.sample_rate
    out_notes = []
    for note in midi_data.instruments[0].notes:
        if note.end > tstart and note.start < tend:
            note.start = max(0, note.start - tstart)
            note.end = min(note.end - tstart, length)
            out_notes.append(note)

    if len(out_notes) == 0:
        return True, None
    midi_data.instruments[0].notes = out_notes
    midi_data.adjust_times([0, length], [0, length])
    return False, midi_data


def main(dummy):
    
    FLAGS.num_signal = 2 * FLAGS.num_signal

    emb_model = None if FLAGS.emb_model_path is None else torch.jit.load(
        FLAGS.emb_model_path).to(FLAGS.device)

    env = lmdb.open(
        FLAGS.output_path,
        map_size=FLAGS.db_size * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    if FLAGS.parser is None:
        audio_files = search_for_audios([FLAGS.input_path])
        audio_files = map(str, audio_files)
        audio_files = map(os.path.abspath, audio_files)
        audio_files = [*audio_files]

        audio_files = [
            f for f in audio_files
            if not any([excl in f for excl in FLAGS.exclude])
        ]
        metadatas = [{"path": audio} for audio in audio_files]
        midi_files = [None] * len(audio_files)
        print(len(audio_files), " files found")

    else:
        audio_files, midi_files, metadatas = get_parser(FLAGS.parser)(
            FLAGS.input_path)

    chunks_buffer, metadatas_buffer = [], []
    midis = []
    cur_index = 0

    # Load BasicPitchPytorch

    if FLAGS.basic_pitch_midi:
        BP = BasicPitchPytorch(sr=FLAGS.sample_rate, device=FLAGS.device)

    # Processing loop
    for i, (file, midi_file, metadata) in enumerate(
            zip(tqdm(audio_files), midi_files, metadatas)):

        try:
            audio = librosa.load(file, sr=FLAGS.sample_rate)[0]
        except:
            print("error loading file : ", file)
            continue

        audio = audio.squeeze()

        if audio.shape[-1] == 0:
            print("Empty file")
            continue

        if FLAGS.normalize:
            audio = normalize_signal(audio)

        # In case no midi_file is used, we can tile the audio file. Otherwise, we need to keep the alignement between midi data and audio.
        if midi_file is None:
            # Pad to a power of 2 if audio is longer than num_signal, tile if audio is too short
            if audio.shape[-1] > FLAGS.num_signal:
                audio = np.pad(
                    audio,
                    (0, FLAGS.num_signal - audio.shape[-1] % FLAGS.num_signal))
            else:
                while audio.shape[-1] < FLAGS.num_signal:
                    audio = np.concatenate([audio, audio])
            
        
        audio = audio[:audio.shape[-1] // FLAGS.num_signal *FLAGS.num_signal]

        # MIDI DATA
        
        if midi_file is not None:
            midi_data = pretty_midi.PrettyMIDI(midi_file)

        elif midi_file is None and FLAGS.basic_pitch_midi:
            midi_data = BP(audio)

        else:
            midi_data = None
        
            
        # Reshape into chunks
        chunks = audio.reshape(-1, FLAGS.num_signal)
        chunk_index = 0

        for j, chunk in enumerate(chunks):
            if midi_data is not None:
                silence_test, midi = get_midi(copy.deepcopy(midi_data), j)
            else:
                midi = None
                silence_test = np.max(
                    abs(chunk)) < 0.05 if FLAGS.cut_silences else False

            # don't process buffer if empty slice
            if silence_test:
                chunk_index += 1
                continue

            midis.append(midi)
            chunks_buffer.append(
                torch.from_numpy(chunk.reshape(1, FLAGS.num_signal)))
            metadatas_buffer.append(metadata)


            if len(chunks_buffer) == FLAGS.batch_size or (
                    j == len(chunks) - 1 and i == len(audio_files) - 1):

                if emb_model is not None:
                    chunks_buffer_torch = (
                        torch.stack(chunks_buffer).squeeze().to(FLAGS.device))
                    z = emb_model.encode(
                        chunks_buffer_torch.reshape(-1, 1, FLAGS.num_signal))
                else:
                    z = [None] * len(chunks_buffer)

                for array, curz, midi, cur_metadata in zip(
                        chunks_buffer, z, midis, metadatas_buffer):

                    ae = AudioExample()

                    if FLAGS.save_waveform:
                        assert array.shape[-1] == FLAGS.num_signal
                        array = (array.cpu().numpy() * (2**15 - 1)).astype(
                            np.int16)
                        ae.put_array("waveform", array, dtype=np.int16)

                    # EMBEDDING
                    if curz is not None:
                        ae.put_array("z", curz.cpu().numpy(), dtype=np.float32)

                    # METADATA
                    cur_metadata["chunk_index"] = chunk_index
                    ae.put_metadata(cur_metadata)

                    # MIDI DATA
                    if midi is not None:
                        ae.put_buffer(key="midi",
                                      b=pickle.dumps(midi),
                                      shape=None)

                    key = f"{cur_index:08d}"

                    with env.begin(write=True) as txn:
                        txn.put(key.encode(), bytes(ae))
                    cur_index += 1

                chunks_buffer, midis, metadatas_buffer = [], [], []
            chunk_index += 1
    env.close()


if __name__ == '__main__':
    app.run(main)
