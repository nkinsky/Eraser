import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from os import path
import scipy.io as sio
from pickle import dump, load
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA as skFastICA
from sklearn.decomposition import PCA as skPCA
import copy

import er_plot_functions as erp
import Placefields as pf
import placefield_stability as pfs
import session_directory as sd
import helpers
from helpers import contiguous_regions
import eraser_reference as err
import freezing_analysis as fa

class DimReduction:
    """"Perform dimensionality reduction on cell activity to pull out coactive ensembles of cells"""
    def __init__(self, mouse: str, arena: str, day: int, bin_size: float = 0.5, nPCs: int = 50,
                 ica_method: str in ['ica_on_cov', 'ica_on_zproj'] = 'ica_on_zproj', **kwargs):
        """Initialize ICA on data. Can also run PCA later if you want.

            :param mouse: str
            :param arena: str
            :param day: int
            :param bin_size: float, seconds, 0.5 = default
            :param nPCs: int, 50 = default
            :param **kwargs: inputs to  sklearn.decomposition.FastICA, e.g. 'tol', 'random_state', etc.
            """

        # Save session info
        self.mouse = mouse
        self.arena = arena
        self.day = day

        # ID working directory
        dir_use = pf.get_dir(mouse, arena, day)
        self.dir_use = Path(dir_use)

        # Load in relevant data
        self.PF = pf.load_pf(mouse, arena, day)
        _, self.freeze_bool = fa.get_freeze_bool(mouse, arena, day)
        self.freeze_ind = np.where(self.freeze_bool)[0]
        md = fa.MotionTuning(mouse, arena, day)
        self.freeze_starts = md.select_events('freeze_onset')
        self.freeze_ends = md.select_events('move_onset')
        # Fix previously calculated occmap
        self.PF.occmap = pf.remake_occmap(self.PF.xBin, self.PF.yBin, self.PF.runoccmap)

        # First bin events
        PSAsmooth, PSAbin = [], []
        self.bin_size = bin_size
        for psa in self.PF.PSAbool_align:
            PSAbin.append(fa.bin_array(psa, int(bin_size * self.PF.sr_image)))  # Create non-overlapping bin array
        self.PSAbin = np.asarray(PSAbin)
        self.PSAbinz = stats.zscore(PSAbin, axis=1)  # Seems to give same results

        # Now calculate covariance matrix for all your cells using binned array
        self.cov_mat = np.cov(self.PSAbin)
        self.cov_matz = np.cov(self.PSAbinz)

        # Run ICA
        self._init_ica_params(**kwargs)
        self._init_PCA_ICA(nPCs=nPCs, ica_method=ica_method)

        # Initialize ensemble testing
        self.pe_rasters = {'pcaica': {'freeze_onset': {}, 'move_onset': {}},
                           'pca': {'freeze_onset': {}, 'move_onset': {}}}
        self.perm_rasters = {'pcaica': {'freeze_onset': {}, 'move_onset': {}},
                             'pca': {'freeze_onset': {}, 'move_onset': {}}}

        try:  # Load in previously calculated tunings
            self.load_sig_tuning()
        except FileNotFoundError:  # if not saved, initialize
            print('No tunings found for this session - run .get_tuning_sig() and .save_sig_tuning()')
            self.sig = {'pcaica': {'freeze_onset': {}, 'move_onset': {}},
                        'pca': {'freeze_onset': {}, 'move_onset': {}}}

    def _init_PCA_ICA(self, nPCs: int, ica_method: str in ['ica_on_cov', 'ica_on_zproj']):
        """Initialize asssemblies based on PCA/ICA method from Lopes dos Santos (2013) and later
        van de Ven (2016) and Trouche (2016)

        :param: nPCs (int) # PCs to look for - should be higher than the expected # of assemblies
        :param: ica_method: (str)
            'ica_on_cov': applies ICA to the covariance matrix of the PCs. Works ok, but in many cases
            does not converge and spits out redundant ICs.
            'ica_on_zproj': applies ICA to the projection of z onto the PCs. Hard to tell, but this is MOST LIKELY
            the intended implementation of the assembly detection method from the above cited literature.
        """
        # Set method type
        self.ica_method = ica_method

        # Run PCA on binned PSA
        self.pca = PCA(self.cov_matz, nPCs)

        # Calculated threshold for significant PCs
        q = self.PSAbinz.shape[1] / self.PSAbinz.shape[0]
        rho2 = 1
        self.pca.lambdamin = rho2 * np.square(1 - np.sqrt(1 / q))
        self.pca.lambdamax = rho2 * np.square(1 + np.sqrt(1 / q))

        # Identify assemblies whose eigenvalues exceed the threshold
        self.nA = np.max(np.where(self.pca.transformer.singular_values_ > self.pca.lambdamax)[0])
        assert self.nA < nPCs, '# assemblies = # PCs, re-run with a larger value for nPCs'

        # Create PCA-reduced activity matrix
        self.psign = self.pca.transformer.components_[0:self.nA].T  # Grab top nA vector weights for neurons
        self.pca.df = self.to_df(self.psign)  # Send PCA weights to dataframe for easy access and plotting
        self.pca.v = self.scale_weights(self.psign)   # Clean up activation weights, make analogous to Dupret method
        self.pca.pmat = self.calc_pmat(self.pca.v)
        self.pca.activations = {'full': self.calc_activations(dr_type='pca', pca_weights_use='v'),
                                'dupret': {'raw': None, 'binned': None, 'binned_z': None}}
        self.zproj = np.matmul(self.psign.T, self.PSAbin)  # Project neurons onto top nA PCAs
        self.cov_zproj = np.cov(self.zproj)  # Get covariance matrix of PCA-projected activity
        self.transformer = skFastICA(random_state=self.random_state, whiten=self.whiten, max_iter=self.max_iter)  # Set up ICA

        # Get ICA weights
        if ica_method == 'ica_on_cov':  # My mistaken first interpretation of method
            self.w_mat = self.transformer.fit_transform(self.cov_zproj)  # Fit it
        elif ica_method == 'ica_on_zproj':  # Most likely method from literature
            self.y = self.transformer.fit_transform(self.zproj.T).T
            if not self.whiten:  # If already
                self.w_mat = self.transformer.components_
            elif self.whiten:  # if whitening performed during ICA, separate out un-mixing matrix
                self.w_mat = np.dot(self.transformer.components_, np.linalg.inv(self.transformer.whitening_))

        # Get final weights of all neurons on PCA/ICA assemblies!
        v = np.matmul(self.psign, self.w_mat)
        self.v = self.scale_weights(v)  # Scale vectors to length 1 and make max weight of each positive

        # Dump into dataframe and get assembly activations over time
        self.df = self.to_df(self.v)
        self.activations = {'full': self.calc_activations('pcaica'),
                            'dupret': {'raw': None, 'binned': None, 'binned_z': None}}

        # Calculate Dupret style activations
        self.pmat = self.calc_pmat(self.v)

        # Initialize activations using raw PSAbool
        self.activations['dupret']['raw'] = self.calc_dupret_activations(psa_use='raw')

        # Initialize significance stats
        self.sig = {'pcaica': None, 'pca': None}
        self._init_sig_dict('pca')
        self._init_sig_dict('pcaica')

    def _init_sig_dict(self, dr_type: str in ['pcaica', 'pca']):
        self.sig[dr_type] = {'full': {'freeze_starts': {'nperm': None, 'pval': None},
                                      'freeze_ends': {'nperm': None, 'pval': None}},
                             'dupret': {'freeze_starts': {'nperm': None, 'pval': None},
                                        'freeze_ends': {'nperm': None, 'pval': None}}}

    def _init_ica_params(self, **kwargs):
        """Sets up parameters for running ICA following PCA."""
        self.random_state = 0
        self.tol = 1e-4
        self.whiten = True
        self.max_iter = 10000
        for key in ('random_state', 'tol', 'whiten', 'max_iter'):
            if key in kwargs:
                setattr(self, key, kwargs[key])

    # def _init_ICA(self, nICs=20):
    #     """Perform ICA"""
    #     # Run ICA on the binned PSA
    #     self.ica = ICA(self.cov_matz, nICs)
    #
    #     # Make into a dataframe for easy plotting
    #     self.ica.df = self.to_df(self.ica.covz_trans)
    #
    #     # Get activations for each IC over time
    #     self.ica.activations = self.calc_activations('ica')
    #     self.ica.nA = nICs

    def _init_PCA(self, nPCs=50):
        """Perform PCA. Will overwrite ICA results"""
        # Run PCA on binned PSA
        self.pca = PCA(self.cov_matz, nPCs)

        # Make into a dataframe for easy plotting
        self.pca.df = self.to_df(self.pca.covz_trans)

        # Clean up activation weights
        self.pca.v = self.scale_weights(self.pca.df)

        # Get activations of each PC
        # self.pca.activations = self.calc_activations('pca')
        self.pca.nA = nPCs

    @staticmethod
    def scale_weights(weights):
        """Scale vmat weights to one and set maximum weight to positive"""

        # Grab basic info about ICs and neurons
        nICs = weights.shape[1]
        valid_bool = np.bitwise_not(np.isnan(weights))
        nneurons_valid = np.sum(valid_bool[:, 0])

        # First, scale weights so that all assemblies have length = 1
        weights_valid = weights[valid_bool].reshape(nneurons_valid, nICs)  # Grab only non-NaN values (rows) of weights mat
        vscale = weights / np.linalg.norm(weights_valid, axis=0)  # Scale full weight mat (including NaNs) by length

        # Next, ensure the highest weight is positive. This shouldn't matter since you later take the outer product
        # of vscale when computing activation patterns, keep here to be consistent with the literature.
        flip_sign = np.greater(np.nanmax(np.abs(vscale), axis=0), np.nanmax(vscale, axis=0))
        vscale_pos = vscale.copy()
        vscale_pos.T[flip_sign] = vscale.T[flip_sign] * -1

        return vscale_pos

    @staticmethod
    def get_titles(dr_type: str in ['pcaica', 'pca', 'ica']):
        """Get plotting titles for different DR types - not sure if this is really used!"""
        if dr_type == 'pcaica':
            dim_red_type = 'PCA/ICA'
            dim_prefix = 'Assembly'
        elif dr_type == 'pca':
            dim_red_type = 'PCA'
            dim_prefix = 'PC'
        elif dr_type == 'ica':
            dim_red_type = 'ICA'
            dim_prefix = 'IC'

        return dim_red_type, dim_prefix

    def plot_assembly_fields(self, assembly: int, ncells_plot: int = 7,
                             dr_type: str in ['pcaica', 'pca', 'ica'] = 'pcaica',
                             tmap_type: str in ['sm', 'us'] = 'sm', isrunning: bool = False,
                             plot_anti: bool = False):

        assert assembly < self.get_nA(dr_type), 'assembly must be less than # of ICs or PCs'

        # Identify top ncells_plot cells that most contribute to the assembly
        df = self.get_df(dr_type)
        assemb_wts = df.iloc[:, assembly + 1]
        bottom_cells = np.argsort(assemb_wts)[-ncells_plot:][::-1]
        top_cells = np.argsort(assemb_wts)[:ncells_plot]

        # Set up plots
        nplots = ncells_plot + 1
        nrows = np.ceil(nplots / 4).astype(int)
        fig, ax = plt.subplots(nrows, 4)
        fig.set_size_inches((nrows*5, 4.75))

        # Plot assembly spatial tuning first
        assemb_pf = self.make_assembly_field(assembly, dr_type, isrunning=isrunning)
        im = ax[0][0].imshow(assemb_pf)
        plt.colorbar(im, ax=ax[0][0])

        _, dim_prefix = self.get_titles(dr_type)
        ax[0][0].set_title(dim_prefix + ' #' + str(assembly))
        ax[0][0].axis('off')
        if tmap_type == 'sm':
            tmap_use = self.PF.tmap_sm
        elif tmap_type == 'us':
            tmap_use = self.PF.tmap_us

        # Determine if assembly has a positive or negative weight
        assemb_max = assemb_pf.reshape(-1).max()
        assemb_min = assemb_pf.reshape(-1).min()
        if np.abs(assemb_max) > np.abs(assemb_min):
            if not plot_anti:
                cells_plot = top_cells
            else:
                cells_plot = bottom_cells
        else:
            if not plot_anti:
                cells_plot = bottom_cells
            else:
                cells_plot = top_cells

        # Plot each cell's tuning also
        for a, cell in zip(ax.reshape(-1)[1:nplots], cells_plot):
            # pf.imshow_nan(self.PF.tmap_sm[cell], ax=a)
            a.imshow(tmap_use[cell])
            a.set_title(f'Cell {cell} wt: {assemb_wts[cell]:0.3g}')
            a.axis('off')
        sns.despine(fig=fig, left=True, bottom=True)

        fig.suptitle(f'{self.mouse} {self.arena}: Day {self.day}')

    def make_occ_map(self, type: str in ['run', 'occ', 'immobile', 'freeze']):
        """Grab specific types of occupancy maps quickly and easily"""

        if type == 'run':
            occ_map_nan = self.PF.runoccmap.copy()
        elif type == 'occ':
            occ_map_nan = pf.remake_occmap(self.PF.xBin, self.PF.yBin, self.PF.runoccmap)
        elif type == 'immobile':  # grab all immobility times
            good_bool = np.bitwise_not(self.PF.isrunning)
            occ_map_nan = pf.remake_occmap(self.PF.xBin, self.PF.yBin, self.PF.runoccmap, good_bool=good_bool)
        elif type == 'freeze':  # grab only extended immobility times that count as actual freezing events
            good_bool = self.freeze_bool
            occ_map_nan = pf.remake_occmap(self.PF.xBin, self.PF.yBin, self.PF.runoccmap, good_bool=good_bool)

        return occ_map_nan

    def make_assembly_field(self, assembly: int, dr_type: str in ['pcaica', 'pca'],
                            act_method: str in ['dupret', 'full'] = 'dupret', isrunning: bool = False):
        """Plot spatial activation of a particular assembly"""
        # Grab occ map to get correct size and shape for 2-d assembly field
        occ_map_nan = self.PF.runoccmap.copy()
        occ_map_nan[occ_map_nan == 0] = np.nan

        # Initialize maps - keep occ_map_check in case you change code later and need a sanity check!
        assemb_pf = np.zeros_like(np.rot90(self.PF.occmap, -1))
        occ_map_check = np.zeros_like(np.rot90(self.PF.occmap, -1))
        nx, ny = assemb_pf.shape

        # Get appropriate activations!
        activations = self.get_activations(dr_type=dr_type, act_method=act_method)

        # Loop through each bin and add up assembly values in that bin
        for xb in range(nx):
            for yb in range(ny):
                in_bin_bool = np.bitwise_and((self.PF.xBin - 1) == xb, (self.PF.yBin - 1) == yb)
                if isrunning:
                    in_bin_bool = np.bitwise_and(in_bin_bool, self.PF.isrunning)
                assemb_pf[xb, yb] = activations[assembly][in_bin_bool].sum()
                occ_map_check[xb, yb] = in_bin_bool.sum()

        assemb_pf = np.rot90(assemb_pf, 1)
        occ_map_check = np.rot90(occ_map_check, 1)
        occ_map_check[occ_map_check == 0] = np.nan

        assemb_pf = assemb_pf * self.PF.occmap / np.nansum(self.PF.occmap.reshape(-1))
        assemb_pf[np.bitwise_or(self.PF.occmap == 0, np.isnan(self.PF.occmap))] = np.nan

        return assemb_pf

    @staticmethod
    def get_sig_plot_tuning(sig: dict, sig_plot: str in ['peaks', 'troughs', 'both'],
                            nassambly: int, alpha: float):
        """Find significant peaks and troughs in freeze onset/offset assembly tuning curves"""
        if 'pval' in sig:
            sig_peaks = [np.where(p < alpha)[0] for p in sig['pval']]
            sig_troughs = [np.where((1 - p) < alpha)[0] for p in sig['pval']]
            if sig_plot == 'peaks':
                sig_bins = sig_peaks
            elif sig_plot == 'troughs':
                sig_bins = sig_troughs
            elif sig_plot == 'both':
                sig_bins = copy.deepcopy(sig_peaks)
                [peaks.extend(troughs) for peaks, troughs in zip(sig_bins, sig_troughs)]

        else:
            sig_bins = [None for _ in range(nassambly)]

        return sig_bins

    def plot_rasters(self, dr_type: str in ['pcaica', 'pca'] = 'pcaica',
                     act_method: str in ['full', 'dupret'] = 'dupret',
                     buffer: int = 6,
                     psa_use: str in ['raw', 'binned', 'binned_z'] = 'raw',
                     events: str in ['freeze_starts', 'freeze_ends'] = 'freeze_starts',
                     alpha: float = 0.01, sig_plot: str in ['peaks', 'troughs', 'both'] = 'peaks',
                     ax=None, **kwargs):
        """Plot rasters of assembly activation in relation to freezing"""

        # Get appropriate activations
        activations = self.get_activations(dr_type, act_method=act_method, psa_use=psa_use)
        nassembly = activations.shape[0]

        # Set up plots
        if ax is None:
            ncomps = activations.shape[0]  # Get # assemblies
            ncols = 5
            nrows = np.ceil(ncomps / ncols).astype(int)
            fig, ax = plt.subplots(nrows, ncols)
            fig.set_size_inches([3 * nrows, 15])
        else:
            nrows = ax.shape[0]
            ncols = 1 if ax.ndim == 1 else ax.shape[1]
            fig = ax.reshape(-1)[0].get_figure()

        # Check if significance tuning calculated for plotting
        assert events in ['freeze_starts', 'freeze_ends']
        assert sig_plot in ['peaks', 'troughs', 'both']
        event_names = 'freeze_onset' if events == 'freeze_starts' else 'move_onset'
        sig_bins = self.get_sig_plot_tuning(self.sig[dr_type][event_names], sig_plot, nassembly, alpha)
        # if 'pval' in self.sig[dr_type][event_names]:
        #     sig_peaks = [np.where(p < alpha)[0] for p in self.sig[dr_type][event_names]['pval']]
        #     sig_troughs = [np.where((1 -p) < alpha)[0] for p in self.sig[dr_type][event_names]['pval']]
        #     if sig_plot == 'peaks':
        #         sig_bins = sig_peaks
        #     elif sig_plot == 'troughs':
        #         sig_bins = sig_troughs
        #     elif sig_plot == 'both':
        #         sig_bins = copy.deepcopy(sig_peaks)
        #         [peaks.extend(troughs) for peaks, troughs in zip(sig_bins, sig_troughs)]
        #
        # else:
        #     sig_bins = [None for _ in activations]

        ensemble_type = 'PCA' if dr_type == 'pca' else 'ICA'

        events_use = getattr(self, events)  # Select events to use

        ntrials = self.freeze_starts.shape[0]
        for ida, (a, act) in enumerate(zip(ax.reshape(-1), activations)):
            # Assemble raster for assembly and calculate mean activation
            assemb_raster = fa.get_PE_raster(act, event_starts=events_use, buffer_sec=(buffer, buffer),
                                          sr_image=self.PF.sr_image)
            act_mean = act.mean()
            # Hack to switch up signs - necessary?
            if act_mean < 0:
                assemb_raster = assemb_raster * -1
                act_mean = act_mean * -1
                suffix = 'f'
            else:
                suffix = ''

            # Figure out which plots to label
            labely = (ida % ncols) == 0  # only label y on left-most plots
            labely2 = ida % ncols == (ncols - 1)  # only labely y2 on right-most plots
            labelx = ida >= ncols * (nrows - 1)
            fa.plot_raster(assemb_raster, cell_id=ida, bs_rate=act_mean, sr_image=self.PF.sr_image, ax=a,
                        y2zero=ntrials / 5, events=events.replace('_', ' '), sig_bins=sig_bins[ida],
                        sig_style='w*', labely=labely, labely2=labely2, labelx=labelx, **kwargs)
            a.set_title(f'{ensemble_type} {ida}{suffix}')

        type_prepend = f' {psa_use.capitalize()}' if act_method == 'dupret' else ''  # Add activation type for dupret
        fig.suptitle(f'{self.mouse} {self.arena} : Day {self.day} {dr_type.upper()}{type_prepend} Activations')

        return ax

    def plot_activations_w_freezing(self, freeze_ensembles: int or list or tuple,
                                    non_freeze_ensembles: int or list or tuple = None,
                                    dr_type: str in ['pcaica', 'pca'] = 'pcaica',
                                    act_method: str in ['full', 'dupret'] = 'dupret',
                                    psa_use: str in ['raw', 'binned', 'binned_z'] = 'raw',
                                    plot_speed: bool = False):
        """Plot activation of freeze (and optionally, non-freeze) ensembles across time with freezing
        epochs overlaid.

        :param freeze_ensembles:
        :param non_freeze_ensembles:
        :param dr_type:
        :param act_method:
        :param psa_use:
        :return:
        """

        # Format inputs properly
        freeze_ensembles = [freeze_ensembles] if isinstance(freeze_ensembles, int) else freeze_ensembles
        non_freeze_ensembles = [non_freeze_ensembles] if isinstance(non_freeze_ensembles, int) else non_freeze_ensembles

        activations = self.get_activations(dr_type=dr_type, act_method=act_method, psa_use=psa_use)

        figf, axover = plt.subplots(figsize=(10, 4))
        t = np.arange(activations.shape[1]) / self.PF.sr_image
        hfall = axover.plot(t, activations[freeze_ensembles].T)  # plot freezing ensembles
        hf, = axover.plot(t, activations[freeze_ensembles].mean(axis=0), 'k-')  # Plot avg freezing ensemble in black

        for freeze_start, freeze_end in zip(self.freeze_starts, self.freeze_ends):
            hfepoch = axover.axvspan(freeze_start, freeze_end)

        if non_freeze_ensembles is not None:
            hnfall = axover.plot(t, activations[non_freeze_ensembles].T, '--')  # plot freezing ensembles
            hnf, = axover.plot(t, activations[non_freeze_ensembles].mean(axis=0),
                               'r--')  # Plo avg non-freezing ensemble in black
            axover.legend((hf, hnf, hfepoch), ('Freeze avg', 'Non-freeze avg', 'Freeze_epoch'))
        else:
            axover.legend((hf, hfepoch), ('Freeze avg', 'Freeze_epoch'))

        return axover

    def plot_PSA_w_activation(self, comp_plot: int, dr_type: str in ['pcaica', 'pca'] = 'pcaica',
                              psa_use: str in ['raw', 'binned', 'binned_z'] = 'raw',
                              plot_freezing=True):
        """Plots all calcium activity with activation of a particular assembly overlaid"""

        # Get appropriate activations
        activations = self.get_activations(dr_type, psa_use=psa_use)

        # First grab activity of a given assembly
        activation = activations[comp_plot]

        # Next, sort PSA according to that assembly
        df = self.get_df(dr_type)
        isort = df[comp_plot].argsort()
        # PSAsort_bin = self.PSAbin[isort[::-1]]
        PSAsort = self.PF.PSAbool_align[isort[::-1]]

        # Now, set up the plot
        fig, ax = plt.subplots()
        fig.set_size_inches([11, 6])
        sns.heatmap(PSAsort, ax=ax, cbar=False)
        _, dim_prefix = self.get_titles(dr_type)
        ax.set_title(f'{dim_prefix} # {comp_plot}')
        # sns.heatmap(PSAsort_bin, ax=ax[1])
        # ax[1].set_title('PSAbin ICA = ' + str(ica_plot))

        # plot freezing times and speed
        if plot_freezing:
            ax.plot(self.freeze_ind, np.ones_like(self.freeze_ind), 'r.', color=[1, 0, 0, 0.5])
            ax.plot(self.freeze_ind, np.ones_like(self.freeze_ind) * (self.PF.PSAbool_align.shape[0] / 2 - 20), '.',
                    color=[1, 0, 0, 0.5])
            ax.plot(self.freeze_ind, np.ones_like(self.freeze_ind) * (self.PF.PSAbool_align.shape[0] - 5), '.',
                    color=[1, 0, 0, 0.5])
            ax.plot(-self.PF.speed_sm * 10 + self.PF.PSAbool_align.shape[0] / 2, '-', color=[0, 0, 1, 0.5])
        # plot activations
        ax.plot(-activation * 100/2 + self.PF.PSAbool_align.shape[0] / 2, '-', color=[0, 1, 0, 0.5])


        # Label things
        ax.set_ylabel('Sorted cell #')
        ax.set_xticklabels([f'{tick/self.PF.sr_image:0.0f}' for tick in ax.get_xticks()])
        ax.set_xlabel('Time (s)')

        return ax

    def calc_speed_corr(self, plot=True, dr_type: str in ['pcaica', 'pca', 'ica'] = 'pcaica'):
        """Correlate activation of each component with speed or project onto freezing to visualize any freeze-related
        assemblies"""

        # Get appropriate activations
        activations = self.get_activations(dr_type, psa_use='raw')

        # Now correlate with speed and multiply by freezing
        freeze_proj, speed_corr = [], []
        for act in activations:
            act = np.asarray(act, dtype=float)
            corr_mat = np.corrcoef(np.stack((act, self.PF.speed_sm), axis=1).T)
            speed_corr.append(corr_mat[0, 1])
            freeze_proj.append((act * self.freeze_bool).sum())

        freeze_proj = np.asarray(freeze_proj)
        speed_corr = np.asarray(speed_corr)

        # Plot
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(speed_corr, freeze_proj)
            ax.set_xlabel('speed correlation')
            ax.set_ylabel('freeze PSAbool projection')
            for idc, (x, y) in enumerate(zip(speed_corr, freeze_proj)):
                ax.text(x, y, str(idc))
            dim_red_type, _ = self.get_titles(dr_type)
            ax.set_title(dim_red_type)
            sns.despine(ax=ax)

        return speed_corr, freeze_proj

    def activation_v_speed(self, dr_type: str in ['pcaica', 'pca'] = 'pcaica'):
        """Scatterplot of all components versus speed"""

        # Get appropriate activations
        activations = self.get_activations(dr_type, psa_use='raw')

        figc, axc = plt.subplots(4, 5)
        figc.set_size_inches([18, 12])
        dim_red_type, dim_prefix = self.get_titles(dr_type)
        for ida, (act, a) in enumerate(zip(activations, axc.reshape(-1))):
            a.scatter(self.PF.speed_sm, act, s=1)
            a.set_title(f'{dim_prefix} #{ida}')
            a.set_xlabel('Speed (cm/s)')
            a.set_ylabel(dim_red_type + ' activation')
        sns.despine(fig=figc)

    @staticmethod
    def to_df(cov_trans):
        """Transforms a reduced covariance matrix array to a dataframe"""
        df = pd.DataFrame(cov_trans)  # Convert to df

        # Add in cell # column - helpful for some plots later
        df["cell"] = [str(_) for _ in np.arange(0, df.shape[0])]
        cols = df.columns.to_list()

        # put cell # in first column
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        return df

    @staticmethod
    def calc_pmat(weight_mat):
        """Calculates projection matrix for each IC"""
        assert weight_mat.shape[0] > weight_mat.shape[1], "Dim 0 of weight mat < Dim 1, check and maybe try transpose"
        pmat = []
        for v in weight_mat.T:  # Step through each weight vector
            ptemp = np.outer(v, v)
            ptemp[np.eye(np.shape(ptemp)[0]).astype(bool)] = 0  # Set diagonal to zero
            pmat.append(ptemp)
        pmat = np.asarray(pmat)

        return pmat

    def get_dupret_activations(self, dr_type: str in ['pcaica', 'pca'] = 'pcaica',
                               psa_use: str in ['raw', 'binnned', 'binnned_z'] = 'raw'):
        """Quickly grabs Dupret style activations and calculates if not already done"""
        # if self.dupret_activations[psa_use] is not None:
        #     print('Grabbing pre-calculated Dupret activations for ' + psa_use + ' data')
        #     dupret_activations = self.dupret_activations[psa_use]
        # else:
        #     print('Calculating Dupret activations using ' + psa_use + ' data')
        #     dupret_activations = self.calc_dupret_activations(psa_use)

        act_dict = self.activations if dr_type == 'pcaica' else self.pca.activations

        if act_dict['dupret'][psa_use] is not None:
            print('Grabbing pre-calculated Dupret activations for ' + psa_use + ' data')
            dupret_activations = act_dict['dupret'][psa_use]
        else:
            dupret_activations = self.calc_dupret_activations(dr_type, psa_use)

        return dupret_activations

    @staticmethod
    def calc_dupret_activation(pmat, psa):
        """Calculate assembly activations in line with Trouche et al (2016). Zeros out diagonal of projection
        matrix so that only CO-ACTIVATIONS of neurons contribute to assembly activation

        :param pmat:  projection matrix for a given assembly, shape ncells x ncells
        :param psa: ncells x nbins array of neural activity
        :return: nbins array of assembly activation strengths.
        """

        act = []
        for psa_t in psa.T:
            act_at_t = np.matmul(np.matmul(psa_t, pmat), psa_t)
            act.append(act_at_t)

        return np.asarray(act)

    def calc_dupret_activations(self, dr_type: str in ['pcaica', 'pca'] = 'pcaica',
                                psa_use: str in ['raw', 'binnned', 'binnned_z'] = 'raw',
                                custom_pmat: None or np.ndarray = None):
        """Calculate assembly activations in line with Trouche et al (2016). Zeros out diagonal of projection
        matrix so that only CO-ACTIVATIONS of neurons contribute to assembly activation"""

        # Select neural activity array to use - should add in gaussian too to compare to Dupret work...
        if psa_use == 'raw':
            psa = self.PF.PSAbool_align
        elif psa_use == 'binned':
            psa = self.PSAbin
        elif psa_use == 'binned_z':
            psa = self.PSAbinz

        # Get appropriate weights for each neuron and activation dictionary
        if custom_pmat is None:
            pmat_use = self.pmat if dr_type == 'pcaica' else self.pca.pmat
            act_dict = self.activations if dr_type == 'pcaica' else self.pca.activations
            pre_calc = act_dict['dupret'][psa_use]
        else:
            pmat_use = custom_pmat
            pre_calc = None

        # Load in existing activations or calculate new ones
        if pre_calc is not None:
            print("Loading pre-calculated Dupret activations calculated from " + psa_use + " calcium activity")
            # return self.dupret_activations[psa_use]
            return pre_calc
        else:
            print("Calculating Dupret activations from " + psa_use + " calcium activity")

            # Calculate activity for each assembly
            d_activations = []
            for p in pmat_use:
                d_activations.append(self.calc_dupret_activation(p, psa))

            # self.dupret_activations[psa_use] = np.asarray(d_activations)
            act_dict['dupret'][psa_use] = np.asarray(d_activations)  # Update dictionary!

            return np.asarray(d_activations)

        # elif dr_type == 'pca':
        #     # Calculate activity for each assembly
        #     d_activations = []
        #     for p in self.pca.pmat:
        #         d_activations.append(self.calc_dupret_activation(p, psa))
        #
        #     return np.asarray(d_activations)

    def calc_activations(self, dr_type: str in ['pcaica', 'pca'], pca_weights_use: str in ['v', 'df'] = 'df'):
        """Pull out activation of each component (IC or PC) over the course of the recording session. Simple
        multiplication of ica with PSAbool. Not consistent with Dupret literature but shows positive AND negative
        activations which are useful to see for QC purposes"""

        # Get appropriate dataframe
        if dr_type == 'pcaica':
            df = self.df
        else:  # Grabs df by default, or use v for something more analagous to dupret method
            df = getattr(getattr(self, dr_type), pca_weights_use)
            if not isinstance(df, pd.DataFrame):  # convert to dataframe if not already there
                df = self.to_df(df)

        # Create array of activations
        activation = []
        for weights in df.iloc[:, df.columns != 'cell'].values.swapaxes(0, 1):
            activation.append(np.matmul(weights, self.PF.PSAbool_align))
        act_array = np.asarray(activation)

        return act_array

    def get_activations(self, dr_type: str in ['pcaica', 'pca'],
                        act_method: str in ['full', 'dupret'] = 'dupret',
                        **kwargs):
        """Quickly grabs pre-calculated activations.  For dupret activations can specifify "raw" (default),
        "binned", or "binned_z in kwargs"""

        # Grab appropriate activations dictionary
        act_dict = self.activations if dr_type == 'pcaica' else self.pca.activations

        if act_method == 'full':
            activations = act_dict['full']
        elif act_method == 'dupret':
            activations = self.get_dupret_activations(dr_type, **kwargs)

        return activations

    def get_nA(self, dr_type: str in ['pcaica', 'pca', 'ica']):

        if dr_type == 'pcaica':
            nA = self.nA
        else:
            nA = getattr(self, dr_type).nA

        return nA

    def get_df(self, dr_type: str in ['pcaica', 'pca', 'ica']):

        if dr_type == 'pcaica':
            df = self.df
        else:
            df = getattr(self, dr_type).df

        return df

    def gen_pe_rasters(self, events='freeze_onset', buffer_sec=(6, 6),
                       dr_type: str in ['pcaica', 'pca'] = 'pcaica'):
        """Generate the rasters for all ensembles and dump them into a dictionary"""

        # Get appropriate event times to use
        assert events in ['freeze_onset', 'move_onset']
        event_starts = self.select_events(events)

        activations = self.get_activations(dr_type=dr_type, act_method='dupret')

        pe_rasters = [fa.get_PE_raster(act, event_starts, buffer_sec=buffer_sec,
                                    sr_image=self.PF.sr_image) for act in activations]

        pe_rasters = np.asarray(pe_rasters)
        self.pe_rasters[dr_type][events] = pe_rasters

        return pe_rasters

    def gen_perm_rasters(self, events='freeze_onset', buffer_sec=(6, 6), nperm=1000,
                         dr_type: str in ['pcaica', 'pca'] = 'pcaica'):
        """Generate shuffled rasters and dump them into a dictionary"""

        # Get appropriate ensembles and event times to use
        assert events in ['freeze_onset', 'move_onset']
        event_starts = self.select_events(events)

        activations = self.get_activations(dr_type=dr_type, act_method='dupret')

        # Loop through each cell and get its chance level raster
        print('generating permuted rasters - may take up to 1 minute')
        perm_rasters = np.asarray([fa.shuffle_raster(act, event_starts, buffer_sec=buffer_sec,
                                                  sr_image=self.PF.sr_image, nperm=nperm)
                                   for act in activations]).swapaxes(0, 1)
        self.perm_rasters[dr_type][events] = perm_rasters

        return perm_rasters

    def get_tuning_sig(self, events='freeze_onset', buffer_sec=(6, 6), nperm=1000,
                       dr_type: str in ['pcaica', 'pca'] = 'pcaica'):
        """This function will calculate significance values by comparing event-centered tuning curves to
        chance (calculated from circular permutation of neural activity) for all ensembles
        :param events:
        :param buffer_sec:
        :return:
        """

        if isinstance(buffer_sec, int):
            buffer_sec = (buffer_sec, buffer_sec)
        # Load in previous tuning
        sig_use = self.sig[dr_type][events]

        calc_tuning = True
        # Check to see if appropriate tuning already run and stored and just use that, otherwise calculate from scratch.
        if 'nperm' in sig_use:
            if sig_use['nperm'] == nperm:
                calc_tuning = False
                pval = sig_use['pval']

        if calc_tuning:
            print('calculating significant tuning for nperm=' + str(nperm))
            # check if both regular and permuted raster are run already!
            pe_rasters, perm_rasters = self.check_rasters_run(events=events,
                                                              buffer_sec=buffer_sec,  nperm=nperm)

            # Now calculate tuning curves and get significance!
            pe_tuning = fa.gen_motion_tuning_curve(pe_rasters)
            perm_tuning = np.asarray([fa.gen_motion_tuning_curve(perm_raster) for perm_raster in perm_rasters])
            pval = (pe_tuning < perm_tuning).sum(axis=0) / nperm

            # Store in class
            self.sig[dr_type][events]['pval'] = pval
            self.sig[dr_type][events]['nperm'] = nperm

            # Save to disk to save time in future
            if hasattr(self, 'dir_use'):
                self.save_sig_tuning()

        return pval

    def get_sig_ensembles(self, events='freeze_onset', buffer_sec=(6, 6), nperm=1000,
                        dr_type: str in ['pcaica', 'pca'] = 'pcaica', alpha=0.01, nbins=3, active_prop=0.25):
        """Find freezing neurons as those which have sig < alpha for nbins (consecutive) or more AND are active on
        at least active_prop of events."""

        # Load in significance values at each spatial bin and re-run things if not already there
        pval = self.get_tuning_sig(events=events, buffer_sec=buffer_sec, nperm=nperm, dr_type=dr_type)

        # Determine if there is significant tuning of < alpha for nbins (consecutive)
        sig_bool = np.asarray([(np.diff(contiguous_regions(p < alpha), axis=1) > nbins).any() for p in pval],
                              dtype=bool)

        # Load in rasters
        pe_rasters = self.gen_pe_rasters(events=events, buffer_sec=buffer_sec)
        nevents = pe_rasters.shape[1]

        # Determine if neurons pass the activity threshold
        nevents_active = pe_rasters.any(axis=2).sum(axis=1)
        active_bool = (nevents_active > active_prop * nevents)

        sig_neurons = np.where(np.bitwise_and(sig_bool, active_bool))[0]

        return sig_neurons

    def check_rasters_run(self, events='freeze_onset', buffer_sec=(6, 6),  nperm=1000,
                          dr_type: str in ['pcaica', 'pca'] = 'pcaica'):
        """ Verifies if you have already created rasters and permuted rasters and checks to make sure they match.

        :param cells:
        :param events:
        :param buffer_sec:
        :param nperm:
        :return:
        """
        # check if both regular and permuted raster are run already!
        pe_rasters = self.pe_rasters[dr_type][events]
        perm_rasters = self.perm_rasters[dr_type][events]
        nbins_use = np.sum([int(buffer_sec[0] * self.PF.sr_image), int(buffer_sec[1] * self.PF.sr_image)])
        if isinstance(pe_rasters, np.ndarray) and isinstance(perm_rasters, np.ndarray):
            ncells, nevents, nbins = pe_rasters.shape
            nperm2, ncells2, nevents2, nbins2 = perm_rasters.shape

            # Make sure you are using the same data format!
            assert ncells == ncells2, '# Cells in data and permuted rasters do not match'
            assert nevents == nevents2, '# events in data and permuted rasters do not match'

            # if different buffer_sec used, re-run full rasters
            if nbins != nbins_use:
                pe_rasters = self.gen_pe_rasters(events=events, buffer_sec=buffer_sec, dr_type=dr_type)

            # if different buffer_sec or nperm used, re-run permuted rasters
            if nbins2 != nbins_use or nperm2 != nperm:
                perm_rasters = self.gen_perm_rasters(events=events, buffer_sec=buffer_sec, nperm=nperm,
                                                     dr_type=dr_type)

        elif not isinstance(pe_rasters, np.ndarray):
            pe_rasters = self.gen_pe_rasters(events=events, buffer_sec=buffer_sec, dr_type=dr_type)
            if not isinstance(perm_rasters, np.ndarray):
                perm_rasters = self.gen_perm_rasters(events=events, buffer_sec=buffer_sec, nperm=nperm,
                                                     dr_type=dr_type)
            else:
                pe_rasters, perm_rasters = self.check_rasters_run(events=events, buffer_sec=buffer_sec, nperm=nperm,
                                                                  dr_type=dr_type)

        return pe_rasters, perm_rasters

    def select_events(self, events):
        """Quickly get the appropriate cells and event times to use"""

        # Get appropriate events
        if events in ['freeze_onset', 'freeze_starts']:
            event_starts = self.freeze_starts
        elif events in ['move_onset', 'freeze_ends']:
            event_starts = self.freeze_ends

        return event_starts

    def save_sig_tuning(self):
        """Saves any significant tuned ensemble tuning significance data"""
        with open(self.dir_use / "sig_ensemble_motion_tuning.pkl", 'wb') as f:
            dump(self.sig, f)

    def load_sig_tuning(self):
        """Loads any previously calculated ensemble tuning significance data"""
        with open(self.dir_use / "sig_ensemble_motion_tuning.pkl", 'rb') as f:
            self.sig = load(f)


class DimReductionReg(DimReduction):
    def __init__(self, mouse: str, base_arena: str, base_day: int, reg_arena: str, reg_day: str, bin_size: float = 0.5,
                 nPCs: int = 50, ica_method: str in ['ica_on_cov', 'ica_on_zproj'] = 'ica_on_zproj', **kwargs):
        """

        :param mouse:
        :param base_arena: Arena/Day to start with
        :param base_day:
        :param reg_arena: Arena/Day to register to base day
        :param reg_day:
        :param bin_size:
        :param nPCs:
        :param ica_method:
        :param kwargs:
        """

        self.mouse = mouse
        self.day = reg_day
        self.arena = reg_arena
        self.base_day = base_day
        self.base_arena = base_arena

        # Create a DimReduction object for each day
        self.DRbase = DimReduction(mouse, base_arena, base_day, bin_size, nPCs, ica_method, **kwargs)
        self.DRreg = DimReduction(mouse, reg_arena, reg_day, bin_size, nPCs, ica_method, **kwargs)

        # Now register across days
        neuron_map = pfs.get_neuronmap(mouse, base_arena, base_day, reg_arena, reg_day, batch_map_use=True)
        self.neuron_map = neuron_map
        nneuronsr = self.DRreg.df.shape[0]
        nics = self.DRbase.df.shape[1] - 1
        valid_map_bool = neuron_map >= 0
        reg_weights_full = np.ones((nneuronsr, nics)) * np.nan
        for weights_reg, weights_base in zip(reg_weights_full.T, self.DRbase.df.values[:, 1:].T.copy()):
            weights_reg[neuron_map[valid_map_bool]] = weights_base[valid_map_bool]

        # Now scale registered weights
        self.v = self.DRreg.scale_weights(reg_weights_full)
        self.df = self.DRreg.to_df(self.v)
        self.pmat = self.DRreg.calc_pmat(self.v)
        self.pmat[np.isnan(self.pmat)] = 0  # Set nans to 0, otherwise activations will all come out as NaN later on
        # self.nics = self.DRreg.df.shape[1] - 1
        self.nics = self.df.shape[1] - 1
        # Pull over freezing times and neural activity from registered day
        self.PF = self.DRreg.PF
        self.freeze_starts = self.DRreg.freeze_starts
        self.freeze_ends = self.DRreg.freeze_ends

        # Calculate activations
        self.activations = {'full': self.calc_activations('pcaica'),
                            'dupret': {'raw': None, 'binned': None, 'binned_z': None}}

        # Initialize ensemble testing
        self.pe_rasters = {'pcaica': {'freeze_onset': {}, 'move_onset': {}},
                           'pca': {'freeze_onset': {}, 'move_onset': {}}}
        self.perm_rasters = {'pcaica': {'freeze_onset': {}, 'move_onset': {}},
                             'pca': {'freeze_onset': {}, 'move_onset': {}}}
        self.sig = {'pcaica': {'freeze_onset': {}, 'move_onset': {}},
                    'pca': {'freeze_onset': {}, 'move_onset': {}}}

    def plot_cov_across_days(self, neurons: str in ['freeze_onset', 'move_onset', 'all'] or np.ndarray = 'freeze_onset',
                             label: str = "", keep_silent: bool = False, ax=None, **kwargs):
        """Plot covariance matrix for neurons in question across days - needs QC.
        NRK - add in silent (off) cells as an option"""

        # plot covariance matrix across days
        MDbase = fa.MotionTuning(self.mouse, self.base_arena, self.base_day)

        assert (isinstance(neurons, str) and neurons in ['freeze_onset', 'move_onset', 'all']) or isinstance(neurons, np.ndarray)
        if isinstance(neurons, str) and neurons in ['freeze_onset', 'move_onset']:
            sig_neurons = MDbase.get_sig_neurons(events=neurons, buffer_sec=(6, 6))
            neuron_label = neurons.capitalize().replace('_', ' ')
        elif isinstance(neurons, str) and neurons == 'all':
            sig_neurons = np.arange(self.DRbase.PF.nneurons)
            neuron_label = neurons.capitalize()
        else:
            sig_neurons = neurons
            neuron_label = label if label != "" else "Custom"

        sig_neurons_reg = self.neuron_map[sig_neurons]
        sigbool = sig_neurons_reg > -1
        sig_neurons_reg = sig_neurons_reg[sigbool]
        sig_neurons_base = sig_neurons[sigbool]

        if not keep_silent:
            covz_base = self.DRbase.cov_matz[sig_neurons_base][:, sig_neurons_base]
            covz_reg = self.DRreg.cov_matz[sig_neurons_reg][:, sig_neurons_reg]
        elif keep_silent:
            covz_base = self.DRbase.cov_matz[sig_neurons][:, sig_neurons]
            covz_reg = np.zeros_like(covz_base)
            covz_reg[np.outer(sigbool, sigbool)] = self.DRreg.cov_matz[sig_neurons_reg][:, sig_neurons_reg].reshape(-1)

        covz_comb = np.tril(covz_base, -1) + np.triu(covz_reg, 1)

        assert ax is None or isinstance(ax, plt.Axes)
        if ax is None:
            figc, axc = plt.subplots()
        else:
            axc = ax
        sns.heatmap(covz_comb, ax=axc, cbar_kws=dict(use_gridspec=False, location="left"), **kwargs)
        axc.set_xlabel(self.base_arena + ' Day ' + str(self.base_day) + ' (Base)')
        axc.set_xticks([])
        axc.set_yticks([])
        secaxy = axc.secondary_yaxis('right')
        secaxy.set_yticks([])
        secaxy.set_ylabel(self.arena + ' Day ' + str(self.day) + ' (Reg)')
        axc.set_title(self.mouse + ' Cov Mat: ' + neuron_label + ' Cells')

        return axc, covz_comb

    def plot_reg_rasters(self, dr_type: str in ['pcaica', 'pca'] = 'pcaica',
                         act_method: str in ['full', 'dupret'] = 'dupret',
                         buffer_sec: int = 6,
                         psa_use: str in ['raw', 'binned', 'binned_z'] = 'raw',
                         **kwargs):
        ax = self.plot_rasters(dr_type, act_method, buffer_sec, psa_use, **kwargs)
        fig = ax.reshape(-1)[0].get_figure()
        type_prepend = f' {psa_use.capitalize()}' if act_method == 'dupret' else ''  # Add activation type for dupret
        fig.suptitle(f'{self.mouse} {self.arena} : Day {self.day} {dr_type.upper()}{type_prepend} Activations\n '
                     f'From Base Session: {self.base_arena} Day {self.base_day}')

        return fig

    def plot_rasters_across_days(self, dr_type: str in ['pcaica', 'pca'] = 'pcaica',
                                 act_method: str in ['full', 'dupret'] = 'dupret',
                                 psa_use: str in ['raw', 'binned', 'binned_z'] = 'raw',
                                 buffer_sec: int = 6, sig_plot: str in ['peaks', 'troughs', 'both'] = 'peaks',
                                 plot_speed_corr: bool = True, plot_freeze_ends: bool = True,
                                 y2scale=0.1, alpha: float = 0.01,
                                 **kwargs):
        """Plot all your freeze-related ensemble data across two sessions. Does not plot well in IDE,
        notebook is ideal to allow for easy vertical scrolling."""

        def label_columns_w_day(axtop):
            """Add 'Base Session' and 'Reg Session' to each row of plots"""
            for title_use, a in zip(('Base Session', 'Reg Session'), axtop):
                a.set_title(title_use + '\n' + a.get_title())

        # Set up plots
        ncols = 2 * (1 + plot_speed_corr + plot_freeze_ends)
        fig, ax = plt.subplots(self.nics, ncols, figsize=(ncols*2, self.nics*2))

        # Order things chronologically
        days = np.array([-2, -1, 0, 4, 1, 2, 7])
        if np.where(self.day == days)[0][0] > np.where(self.base_day == days)[0][0]:
            base_col, reg_col = 0, 1
        else:
            base_col, reg_col = 1, 0
        base_reg_col = [base_col, reg_col]

        # Check if significance tuning calculated for plotting
        assert sig_plot in ['peaks', 'troughs', 'both']

        # Plot freeze-start rasters
        for ids, session in enumerate([self.DRbase, self]):
            session.get_tuning_sig('freeze_onset', buffer_sec, nperm=1000, dr_type='pcaica')
            col_plot = base_reg_col[ids]
            session.plot_rasters(dr_type, act_method, buffer_sec, psa_use, events='freeze_starts', ax=ax[:, col_plot],
                                 y2scale=y2scale, alpha=alpha,  **kwargs)
            # self.DRbase.plot_rasters(dr_type, act_method, buffer_sec, psa_use, events='freeze_starts', ax=ax[:, base_col],
            #                          y2scale=y2scale, **kwargs)
            # self.plot_rasters(dr_type, act_method, buffer_sec, psa_use, events='freeze_starts', ax=ax[:, reg_col],
            #               y2scale=y2scale, **kwargs)
        label_columns_w_day(ax[0][[base_col, reg_col]])

        # Plot speed v activity x-correlations
        if plot_speed_corr:
            for col_use, DR in zip((base_col + 2, reg_col + 2), (self.DRbase, self.DRreg)):
                activations = DR.get_activations(dr_type=dr_type, act_method=act_method)
                speed = DR.PF.speed_sm
                for a, act in zip(ax[:, col_use], activations):
                    fa.plot_speed_activity_xcorr(act, speed, buffer_sec, self.PF.sr_image, a, '', labelx=False)
                    sns.despine(ax=a)
            label_columns_w_day(ax[0][[2 + base_col, 2 + reg_col]])

        # Plot freeze end rasters
        if plot_freeze_ends:
            for ids, session in enumerate([self.DRbase, self]):
                # sig_bins = self.get_sig_plot_tuning(session.sig[dr_type]['move_onset'], sig_plot, self.nics, alpha)
                session.get_tuning_sig('move_onset', (buffer_sec, buffer_sec), nperm=1000, dr_type='pcaica')
                col_plot = base_reg_col[ids]
                session.plot_rasters(dr_type, act_method, buffer_sec, psa_use, events='freeze_ends',
                                     ax=ax[:, col_plot + 4], y2scale=y2scale, alpha=alpha, **kwargs)
                # self.DRbase.plot_rasters(dr_type, act_method, buffer_sec, psa_use, events='freeze_ends',
                #                          ax=ax[:, base_col + 4], y2scale=y2scale, **kwargs)
                # self.plot_rasters(dr_type, act_method, buffer_sec, psa_use, events='freeze_ends',
                #                   ax=ax[:, reg_col + 4], y2scale=y2scale, **kwargs)
            label_columns_w_day(ax[0][[4 + base_col, 4 + reg_col]])

        # Label top
        type_prepend = f' {psa_use.capitalize()}' if act_method == 'dupret' else ''  # Add activation type for dupret
        fig.suptitle(f'{self.mouse} {self.arena} : Day {self.day} {dr_type.upper()}{type_prepend} Activations\n '
                     f'From Base Session: {self.base_arena} Day {self.base_day}')

        return fig


class PCA:
    def __init__(self, cov_matz: np.ndarray, nPCs: int = 50):
        """Perform PCA on z-scored covariance matrix"""
        # Run PCA on binned PSA
        self.transformer = skPCA(n_components=nPCs)
        self.covz_trans = self.transformer.fit_transform(cov_matz)


# class ICA:
    # def __init__(self, cov_matz: np.ndarray, nICs: int = 20):
    #     """Perform ICA on z-scored covariance matrix"""
    #     # Run ICA on the binned PSA
    #     self.transformer = skFastICA(n_components=nICs, random_state=0)
    #     self.covz_trans = self.transformer.fit_transform(cov_matz)

