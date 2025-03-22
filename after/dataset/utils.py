import numpy as np


def get_beat_signal(b, len_wave, len_z, sr=24000, zero_value=0):
    if len(b) < 2:
        #print("empty beat")
        return zero_value * np.ones(len_z)
    times = np.linspace(0, len_wave / sr, len_z)
    t_max = times[-1]
    i = 0
    while i < len(b) - 1 and b[i] < t_max:
        i += 1
    b = b[:i]
    minvalue = 0
    id_time_min = 0
    out = []
    true_out = []

    if len(b) < 4:
        #print("empty beat")
        return np.zeros(len(times))
    for i in range(len(b)):
        time = b[i]
        time_prev = b[i - 1] if i > 0 else 0
        delt = time - times

        try:
            id_time_max = np.argmin(delt[delt > 0])
            time_interp = times[id_time_max]
            maxvalue = (time_interp - time_prev) / (time - time_prev)
        except:
            id_time_max = 1
            maxvalue = 1

        out.append(
            np.linspace(minvalue, maxvalue, 1 + id_time_max - id_time_min))

        if i < len(b) - 1:
            minvalue = (times[id_time_max + 1] - time) / (b[i + 1] - time)
            id_time_min = id_time_max + 1

    maxvalue = (times[len_z - 1] - time) / (time - time_prev)
    minvalue = (times[id_time_max] - time) / (time - time_prev)
    id_time_min = id_time_max + 1
    out.append(np.linspace(minvalue, maxvalue, 1 + len_z - id_time_min))

    out = np.concatenate(out)
    out = out[:len(times)]
    if len(out) < len(times):
        out = np.concatenate((out, np.zeros(abs(len(times) - len(out)))))
    return out
