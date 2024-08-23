from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import seaborn as sns
from os import environ

from neuropy.utils.ccg import correlograms
from neuropy.utils.mathutil import contiguous_regions
from neuropy.io.openephysio import get_dat_timestamps, get_lfp_timestamps
from neuropy.io.neuroscopeio import NeuroscopeIO
from neuropy.core.epoch import Epoch

if environ['HOME'] == '/home/nkinsky':
    working_dir = Path('/Users/nkinsky/Documents/UM/Working/Anisomycin/Recording_Rats/Wedge')
elif environ['HOME'] == '/Users/kimqi':
    working_dir = Path('/media/kimqi/BK/Data/Anisomycin/Recording_Rats/Creampuff')


def get_cluster_info(folder_use, keep_good_only=True, working_dir=working_dir):
    """Grabs cluster info and returns a dataframe, keeping only good units (those with an entry in their 'q' field)
    by default"""
    spyk_circ_dir = working_dir / folder_use / 'spyk-circ'
    unit_dir = sorted(spyk_circ_dir.glob("**/spike_times.npy"))[0].parent

    clu_info_file = unit_dir / "cluster_info.tsv"

    clu_info_df = pd.read_csv(clu_info_file, sep="\t")

    if keep_good_only and 'q' in clu_info_df.keys():
        return clu_info_df[~np.isnan(clu_info_df["q"])]
    else:
        return clu_info_df


def load_events_csv(events_file_dir, start_time, working_dir=working_dir):
    """Loads epochs events file delineating start/end of each recording. """
    events_file_dir = working_dir / events_file_dir
    events_file = sorted(Path(events_file_dir).glob("*_events_full.csv"))[0]
    events_df = pd.read_csv(events_file)
    day_offset = pd.tseries.offsets.DateOffset(days=(start_time[0].date() -
                                                     pd.to_datetime(events_df['start']).dt.date[0]).days)
    starts = pd.to_datetime(events_df['start']) + day_offset
    stops = pd.to_datetime(events_df['stop']) + day_offset
    epochs = pd.DataFrame({'start': starts, 'stop': stops, 'label': events_df['label']})

    return epochs


def epochs_to_neuroscope(session_folder, timestamps=None, SR=30000, working_dir=working_dir):
    """Export event epochs to neuroscope format"""

    if timestamps is None:
        timestamps = get_dat_timestamps(working_dir / session_folder)

    epochs = load_events_csv(working_dir / session_folder, timestamps.iloc[0])

    ts = timestamps.squeeze()  # make timestamps into a series

    starts, stops = [], []
    for epoch in epochs['label']:
        epoch_use = epochs[epochs['label'] == epoch]
        start_sec = ts.searchsorted(epoch_use['start']).squeeze() / SR
        stop_sec = ts.searchsorted(epoch_use['stop']).squeeze() / SR
        starts.append(start_sec)
        stops.append(stop_sec)

    epochs_out = pd.DataFrame({"start": starts, "stop": stops, "label": epochs["label"]})
    epochs_out = Epoch(epochs_out)
    recinfo = NeuroscopeIO(sorted((working_dir / session_folder).glob("*.xml"))[0])
    recinfo.write_epochs(epochs_out)

    return epochs_out


def get_single_units(folder_use, SR=30000, keep_separate=False, working_dir=working_dir):
    """Get single unit activity from each session after curation in phy"""
    spyk_circ_dir = working_dir / folder_use / 'spyk-circ'
    unit_dir = sorted(spyk_circ_dir.glob("**/spike_times.npy"))[0].parent

    clu_file = unit_dir / "spike_clusters.npy"
    time_file = unit_dir / "spike_times.npy"

    clu_id = np.load(clu_file)
    sp_id = np.load(time_file)

    if not keep_separate:
        spike_times = sp_id / SR

        return spike_times, clu_id
    else:
        spike_times = []
        for clu in np.unique(clu_id):
            spike_times.append(np.array(sp_id[clu_id == clu] / SR))

        return spike_times, clu_id


def calc_burst_index(sp_times, burst_thresh_ms=8, nspikes_min=3):
    """3 or more spikes fired with less than 8ms ISI per Mizuseki and Buzsaki (2013)"""

    nspikes_total = len(sp_times)
    isi = np.diff(sp_times)  # calculate iSI
    isi_thresh = isi < burst_thresh_ms / 1000  # ID spikes with ISI < thresh
    cand_burst_events = contiguous_regions(isi_thresh)  # ID start and end of candidate burst events
    nspikes_per_event = np.diff(cand_burst_events, axis=1).reshape(-1) + 1  # count up spikes in each event
    burst_event_bool = nspikes_per_event >= nspikes_min  # ID events with >= nspikes_min spikes
    nburst_spikes = nspikes_per_event[burst_event_bool].sum()  # Grab # spikes in all burst

    return nburst_spikes / nspikes_total


def calc_burst_ind_by_epoch(session_folder, timestamps=None, SR=30000, working_dir=working_dir):
    """Calculate burst index by epoch"""
    # Load timestamps if not done already
    if timestamps is None:
        timestamps = get_dat_timestamps(working_dir / session_folder)

    spike_times, clu_ids = get_single_units(session_folder, keep_separate=False)

    clu_info = get_cluster_info(session_folder, keep_good_only=True)
    good_clu = clu_info['cluster_id']
    epochs = load_events_csv(working_dir / session_folder, timestamps.iloc[0])

    ts = timestamps.squeeze()  # Make timestamps into a series

    BI_by_epoch_all = []
    for clu_id in good_clu:
        sp_times_use = spike_times[clu_id == clu_ids]
        BI_by_epoch = []
        epochs_used = []
        for epoch in epochs['label']:
            epoch_use = epochs[epochs['label'] == epoch]
            start_sec = ts.searchsorted(epoch_use['start']) / SR
            stop_sec = ts.searchsorted(epoch_use['stop']) / SR
            if start_sec != stop_sec:  # Skip the below for injection or any epoch with no data!
                epoch_bool = np.bitwise_and(sp_times_use > start_sec, sp_times_use < stop_sec)
                BI_by_epoch.append(calc_burst_index(sp_times_use[epoch_bool]))
                epochs_used.append(epoch)
        #             else:
        #                 print(f'No data found for {epoch} epoch. Skipping')
        BI_by_epoch_all.append(BI_by_epoch)
    BI_by_epoch_all = np.array(BI_by_epoch_all)
    assert BI_by_epoch_all.shape[1] == len(epochs_used), 'epochs data does not match BI data, check code'

    return epochs_used, BI_by_epoch_all


def calc_firing_rate(spike_times):
    """Super rough firing rate calculation"""
    start_time = spike_times.min()
    end_time = spike_times.max()
    epoch_duration = end_time - start_time

    return len(spike_times) / epoch_duration


def calc_firing_rate_by_epoch(session_folder, timestamps=None, SR=30000, combine_units=False, working_dir=working_dir):
    """Calculate burst index by epoch"""
    # Load timestamps if not done already
    if timestamps is None:
        # timestamps = get_dat_timestamps(working_dir / session_folder)
        timestamps = get_lfp_timestamps(working_dir / session_folder)

    spike_times, clu_ids = get_single_units(session_folder, keep_separate=False)

    clu_info = get_cluster_info(session_folder, keep_good_only=True)
    good_clu = clu_info['cluster_id']
    epochs = load_events_csv(working_dir / session_folder, timestamps.iloc[0])

    if combine_units:  # Set all ids to first good unit
        clu_ids = good_clu[0]

    ts = timestamps.squeeze()  # Make timestamps into a series

    FR_by_epoch_all = []
    for clu_id in good_clu:
        sp_times_use = spike_times[clu_id == clu_ids]
        FR_by_epoch = []
        epochs_used = []
        for epoch in epochs['label']:
            epoch_use = epochs[epochs['label'] == epoch]
            start_sec = ts.searchsorted(epoch_use['start']) / SR
            stop_sec = ts.searchsorted(epoch_use['stop']) / SR
            if start_sec != stop_sec:  # Skip the below for injection or any epoch with no data!
                epoch_bool = np.bitwise_and(sp_times_use > start_sec, sp_times_use < stop_sec)
                FR_by_epoch.append(calc_firing_rate(sp_times_use[epoch_bool]))
                epochs_used.append(epoch)
        #             else:
        #                 print(f'No data found for {epoch} epoch. Skipping')
        FR_by_epoch_all.append(FR_by_epoch)
    FR_by_epoch_all = np.array(FR_by_epoch_all)
    assert FR_by_epoch_all.shape[1] == len(epochs_used), 'epochs data does not match BI data, check code'

    return epochs_used, FR_by_epoch_all


def calc_ccg_by_epoch(session_folder, timestamps=None, SR=30000, window_size=0.5, bin_size=0.001, combine_units=False,
                      working_dir=working_dir):
    """Calculate CCG between all cells across each epoch of the recording"""
    # Load timestamps if not done already
    if timestamps is None:
        timestamps = get_dat_timestamps(working_dir / session_folder)

    spike_times, clu_ids = get_single_units(session_folder, keep_separate=False, working_dir=working_dir)
    #     clu_list = np.unique(clu_ids)
    #     n_units = len(clu_list)

    clu_info = get_cluster_info(session_folder, keep_good_only=True, working_dir=working_dir)
    good_clu = clu_info['cluster_id'].values
    n_units = len(good_clu)

    good_clu_bool = [clu_id in good_clu for clu_id in clu_ids]
    if combine_units:  # Set all ids to first good unit
        clu_ids[good_clu_bool] = good_clu[0]
    epochs = load_events_csv(working_dir / session_folder, timestamps.iloc[0])

    ts = timestamps.squeeze()  # Make timestamps into a series

    corr_by_epoch = []
    epochs_used = []
    # time_bins = np.arange(0, int(window_size/bin_size) + 1)
    time_bins = np.linspace(-window_size/2, window_size/2, int(window_size/bin_size) + 1)
    for epoch in epochs['label']:
        print(f'Calculating CCGs for epoch {epoch}')
        epoch_use = epochs[epochs['label'] == epoch]
        start_sec = ts.searchsorted(epoch_use['start'].dt.tz_localize(ts.iloc[0].tz))/SR
        stop_sec = ts.searchsorted(epoch_use['stop'].dt.tz_localize(ts.iloc[0].tz))/SR
        if start_sec != stop_sec:  # Skip the below for injection or any epoch with no data!
            epoch_bool = np.bitwise_and(spike_times > start_sec, spike_times < stop_sec)
            epoch_bool = np.bitwise_and(epoch_bool, good_clu_bool)

#             print(f'# Spikes for units in {epoch} epoch:')
#             print([np.sum(clu == clu_ids[epoch_bool]) for clu in np.unique(clu_ids[epoch_bool])])

            clu_active = np.unique(clu_ids[epoch_bool])
            corr_temp = np.ones((n_units, n_units, len(time_bins)))*np.nan  # pre-allocate
            active_clu_bool = np.array([clu in clu_active for clu in good_clu])  # find all neurons with a spike in this epoch

            corr_temp2 = correlograms(spike_times[epoch_bool], clu_ids[epoch_bool],
                                     bin_size=bin_size, sample_rate=SR, window_size=window_size)

            nactive = active_clu_bool.sum()
            assert nactive == len(clu_active)

            # Dump active neurons into the right spot in the ccg array
            corr_temp[np.outer(active_clu_bool, active_clu_bool), :] = corr_temp2.reshape(nactive*nactive, -1)

            corr_by_epoch.append(corr_temp)
            epochs_used.append(epoch)

    corr_by_epoch = np.array(corr_by_epoch)

    return corr_by_epoch, time_bins, epochs_used


def plot_ccg_by_epoch(corr_by_epoch_use, time_bins_use, epochs_use, session_name, xlims=None, ax=None):
    """Plot ccgs by epoch. Takes output of calc_ccg_by_epoch"""

    # Set up plots
    bin_width = np.diff(time_bins_use).mean() * 0.95
    ncells, nepochs = corr_by_epoch_use.shape[1], corr_by_epoch_use.shape[0]

    if ax is None:
        fig, ax = plt.subplots(ncells, nepochs, figsize=(2.5 * nepochs, 2 * ncells), squeeze=False)
        # Label figure with session name
        fig.suptitle(session_name)
    else:
        fig = ax.reshape(-1)[0].figure

    # Plot everything
    for ae, corr_epoch in zip(ax.T, corr_by_epoch_use):
        for ida, a in enumerate(ae):
            a.bar(time_bins_use, corr_epoch[ida, ida], width=bin_width)
            a.set_xlabel('Time (sec)')
            a.set_ylabel('Count')

    # Label each plot on top row
    for atop, epoch in zip(ax[0, :], epochs_use):
        atop.set_title(epoch)

    if xlims is not None:
        [a.set_xlim(xlims) for a in ax.reshape(-1)]

    # Label cells in a rough way
    [aside.set_ylabel(f'Cell #{ida}') for ida, aside in enumerate(ax[:, 0])];

    # Label x and y axes


def bin_spikes2(sp_times, bin_size_sec=10):
    """Much faster than the above!!!"""
    time_bins = np.arange(0, np.ceil(sp_times.max()) + bin_size_sec, bin_size_sec)

    binned_rate = np.histogram(sp_times, bins=time_bins)[0] / bin_size_sec

    return time_bins, np.array(binned_rate)


if __name__ == "__main__":
    working_dir = Path('/data3/Anisomycin/Recording_Rats/Creampuff')
    ani_timestamps = get_lfp_timestamps(working_dir / "2024_07_17_Anisomycin")
    calc_ccg_by_epoch("2024_07_17_Anisomycin", timestamps=ani_timestamps, working_dir=working_dir)