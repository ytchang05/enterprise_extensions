# -*- coding: utf-8 -*-

import functools
import inspect
from collections import OrderedDict

import numpy as np
from enterprise import constants as const
from enterprise.signals import (deterministic_signals, gp_signals, parameter,
                                selections, signal_base, white_signals)
from enterprise.signals.signal_base import LogLikelihood

from enterprise_extensions import deterministic
from enterprise_extensions import dropout as do
from enterprise_extensions import model_utils
from enterprise_extensions.blocks import (bwm_block, bwm_sglpsr_block,
                                          chromatic_noise_block,
                                          common_red_noise_block,
                                          dm_noise_block, red_noise_block,
                                          white_noise_block)
from enterprise_extensions.chromatic import (dmx_signal, dm_annual_signal,
                                             dm_exponential_dip,
                                             dm_exponential_cusp,
                                             dm_dual_exp_cusp)
from enterprise_extensions.chromatic.solar_wind import solar_wind_block
from enterprise_extensions.timing import timing_block

# from enterprise.signals.signal_base import LookupLikelihood


def model_singlepsr_noise(psr, psr_model=False,
                          is_wideband=False, use_dmdata=False,
                          extra_sigs=None, dense_like=False,
                          noise_dict=None, shared=None,
                          tm=None, white_noise=None,
                          fact_like=None, red_noise=None,
                          dm=None, sw=None, chrom=None):

    """
    Single pulsar noise model.

    This function is set up so all noise block kwargs are optional; specify
    only the kwargs necessary, as unprovided kwargs will use a given noise
    block's defaults. There is also maximum flexibility in providing
    higher-level kwarg defaults that can be overridden by lower-level kwargs.
    If a block has any provided kwargs, its toggle defaults to True; empty
    blocks' toggles default to False, with the exception of the red_noise
    toggle always defaulting to True unless specified False.

    :param psr: enterprise pulsar object
    :param psr_model: Return the enterprise model instantiated on the pulsar
        rather than an instantiated PTA object, i.e. model(psr) rather than
        PTA(model(psr)).
    :param is_wideband: whether input TOAs are wideband TOAs; will exclude
       ecorr from the white noise model
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
           is_wideband
    :param extra_sigs: Any additional `enterprise` signals to be added to the
        model.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param noise_dict: dictionary of noise parameters
    :param shared: dictionary of any noise parameters that may be shared
        between multiple noise blocks; any provided shared kwargs will be the
        default in a given noise block (if used in the block) unless
        overridden in the block's kwarg; examples include:
        - coefficients: explicitly include latent coefficients in model
        - components: number of modes in Fourier domain processes
        - gamma_val: spectral index to fix
        - logmin: specify lower prior for psd
        - logmax: specify upper prior for psd
        - prior: 'log-uniform' or 'uniform'
        - psd: red noise psd model
        - Tspan: time baseline used to determine Fourier GP frequencies
    :param tm: dictionary of timing model kwargs; includes:
        - toggle: vary the timing model parameters
        - dmjump_var:
        - linear: vary the timing model in the linear approximation
        - marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
        - param_list: an explicit list of timing model parameters to vary
        - svd: boolean for svd-stabilised timing model design matrix
        - norm: normalize the timing model, or provide custom normalization
    :param white_noise: dictionary of white noise kwargs; includes:
        - toggle: vary the white noise parameters
        - wb_efac_sigma
        - ng_twg_setup
    :param fact_like: dictionary of common red noise kwargs; includes:
        - toggle: run a factorized likelihood analyis Boolean
        - components: number of modes in Fourier domain for a common
           process in a factorized likelihood calculation.
    :param red_noise: dictionary of red noise kwargs; includes:
        - toggle: include red noise in the model (always default True)
    :param dm: dictionary of DM noise kwargs; includes:
        - vary: whether to vary the DM model or use constant values
        - gp: dictionary of gaussian process DM noise kwargs; includes:
            - toggle: include DMGP
            - gp_kernel: GP kernel type to use, ['diag','nondiag']
            - psd: power-spectral density of DM variations
            - nondiag_kernel: type of time-domain DM GP kernel
            - dt: time-scale for DM linear interpolation basis (days)
            - df: frequency-scale for DM linear interpolation basis (MHz)
            - gamma_val: spectral index of power-law DM variations
        - dmx: dictionary of DMX noise kwargs; includes:
            - toggle: include DMX
            - dmx_data: supply the DMX data from par files
        - annual: dictionary of annual DM noise kwargs; includes: "toggle"
        - expdip, cusp, dual_cusp: dictionary of DM event noise kwargs; each
            behaves the same, and includes:
            - toggle: include DM event noise
            - name: name of event sequence
            - events: list of dictionaries of event kwargs; possible kwargs are
                below, however any of these can also be specified at the same level
                as the toggle above, to be the default for all events of this type:
                - idx: chromatic index of DM event (for dual_cusp, idx1 and idx2)
                - tmin: sampling minimum of DM event epoch
                - tmax: sampling maximum of DM event epoch
                - sign: set the sign parameter for event
                - sym: make event symmetric (for cusp and dual_cusp)
                - name: name of event
    :param sw: dictionary of solar wind noise kwargs; includes:
        - toggle: use the deterministic solar wind model
        - include_swgp: add a Gaussian process perturbation to the deterministic
        solar wind model (default False)
        - swgp_prior: prior is currently set automatically
        - swgp_basis: ['powerlaw', 'periodic', 'sq_exp']
    :param chrom: dictionary of chromatic noise kwargs; includes:
        - toggle: include general chromatic noise
        - vary: whether to vary the chromatic model or use constant values
        - gp_kernel: GP kernel type to use, ['diag','nondiag']
        - psd: power-spectral density of chromatic noise
            ['powerlaw','tprocess','free_spectrum']
        - idx: frequency scaling of chromatic noise. use 'vary' to vary
            between [2.5, 5]
        - nondiag_kernel: type of 'nondiag' time-domain chrom GP kernel to use
            ['periodic', 'sq_exp','periodic_rfband', 'sq_exp_rfband']
        - include_quadratic: whether to add a quadratic chromatic term
        - dt: time-scale for chromatic linear interpolation basis (days)
        - df: frequency-scale for chromatic linear interpolation basis (MHz)

    :return s: single pulsar noise model

    """

    # TODO: add **kwargs and convert old kwargs for backward compatibility

    # default kwarg dicts to empty dicts
    shared = shared or {}
    tm = tm or {}
    white_noise = white_noise or {}
    fact_like = fact_like or {}
    red_noise = red_noise or {}
    dm = dm or {}
    sw = sw or {}
    chrom = chrom or {}

    # timing model
    tm_settings = {k: tm.pop(k, False) for k in ("dmjump_var", "linear", "marg")}
    if not (tm and tm.pop("toggle", True)):
        if is_wideband and use_dmdata:

            if tm_settings["dmjump_var"]:
                dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
            else:
                dmjump = parameter.Constant()

            if white_noise and white_noise.get("toggle", True):
                if white_noise.get("ng_twg_setup"):
                    default_wb_efac_sigma = inspect.signature(white_noise_block).parameters["wb_efac_sigma"].default
                    dmefac = parameter.Normal(1.0, white_noise.get("wb_efac_sigma", default_wb_efac_sigma))
                else:
                    dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
                log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
                # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)

            else:
                dmefac = parameter.Constant()
                log10_dmequad = parameter.Constant()
                # dmjump = parameter.Constant()

            s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                               log10_dmequad=log10_dmequad, dmjump=dmjump,
                                               selection=selections.Selection(selections.by_backend),
                                               dmjump_selection=selections.Selection(selections.by_frontend))

        else:
            if tm_settings["marg"]:
                s = gp_signals.MarginalizingTimingModel(**tm)
            else:
                default_coefficients = inspect.signature(gp_signals.TimingModel).parameters["coefficients"].default
                tm["coefficients"] = shared.get("coefficients", default_coefficients)
                s = gp_signals.TimingModel(**tm)

    else:
        # create new attribute for enterprise pulsar object
        psr.tmparams_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
        for key in psr.tmparams_orig:
            psr.tmparams_orig[key] = (psr.t2pulsar[key].val,
                                      psr.t2pulsar[key].err)
        if not tm_settings["linear"]:
            s = timing_block(**tm)
        else:
            raise NotImplementedError("Linear timing model not implemented yet.")

    # white noise vary
    if white_noise and white_noise.get("toggle", True):
        white_noise["vary"] = white_noise.get("vary", True)

    # fact like overrides
    if fact_like.get("toggle") and not fact_like.get("Tspan"):
        raise ValueError("Must Timespan to match amongst all pulsars when doing " +
                         "a factorized likelihood analysis.")
    fact_like["gamma_val"] = fact_like.get("gamma_val", 13./3)

    # default red noise toggle to true
    if not red_noise:
        print("Red noise toggle not specified, defaulting to True.")
        red_noise["toggle"] = True

    # use dm_vary for all dm blocks unless overridden
    if "vary" in dm:
        for key in [k for k in dm if k != "vary"]:
            dm[key]["vary"] = dm[key].get("vary", dm["vary"])

    # set dmx data for specific pulsar
    if dm.get("dmx", {}).get("dmx_data"):
        dm["dmx"]["dmx_data"] = dm["dmx"]["dmx_data"][psr.name]

    # override solar wind block defaults
    sw["ACE_prior"] = sw.get("ACE_prior", True)
    sw["include_gp"] = sw.get("include_gp", False)

    # adding white-noise, and acting on psr objects
    if ('NANOGrav' in psr.flags['pta'] or 'CHIME' in psr.flags['f']) and not is_wideband:
        white_noise["inc_ecorr"] = white_noise.get("inc_ecorr", True)

    blocks = {
        "white_noise": white_noise_block,
        "fact_like": common_red_noise_block,
        "red_noise": red_noise_block,
        "dm.gp": dm_noise_block,
        "dm.dmx": dmx_signal,
        "dm.annual": dm_annual_signal,
        "sw": solar_wind_block,
        "chrom": chromatic_noise_block,
    }

    for name, block in blocks.items():

        # get toggle and kwargs from arguments
        block_kwargs = locals()[name] if not name.startswith("dm.") else dm.get(name[3:], {})

        if block_kwargs and block_kwargs.pop("toggle", True):

            # use shared_kwargs where applicable unless overridden
            for key, value in shared.items():
                if inspect.signature(block).parameters.get(key):
                    block_kwargs[key] = block_kwargs.get(key, value)

            # add noise block
            s += block(**block_kwargs)

    dm_event_blocks = {
        "expdip": dm_exponential_dip,
        "cusp": dm_exponential_cusp,
        "dual_cusp": dm_dual_exp_cusp
    }

    for name, block in dm_event_blocks.items():

        if dm and dm.get(name) and dm[name].pop("toggle", True):

            # enumerate events for naming purposes
            for n, event_kwargs in enumerate(dm[name].get("events", [{}])):

                # use shared kwargs where applicable unless overridden
                for k, v in dm[name].items():
                    if k in ("events", "name"):
                        continue
                    event_kwargs[k] = event_kwargs.get(k, v)

                # use default tmin and tmax unless overridden
                event_kwargs["tmin"] = event_kwargs.get("tmin", psr.toas.min() / const.day)
                event_kwargs["tmax"] = event_kwargs.get("tmax", psr.toas.max() / const.day)

                # format event name
                event_kwargs["name"] = '_'.join([x for x in
                                                 ["dm", name, dm[name].get('name'),
                                                  (event_kwargs.get("name") or str(n+1))]
                                                 if x])

                # add noise block
                s += block(**event_kwargs)

    # extra signals
    if extra_sigs is not None:
        s += extra_sigs

    # return enterprise model
    if psr_model:
        return s

    # set up PTA
    model = s(psr)
    if dense_like:
        pta = signal_base.PTA([model], lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA([model])

    # set white noise parameters
    if noise_dict is not None:
        pta.set_default_params(noise_dict)
    elif not white_noise or (is_wideband and use_dmdata):
        print('No noise dictionary provided!...')

    return pta


def model_1(psrs, psd='powerlaw', noisedict=None, white_vary=False,
            components=30, upper_limit=False, bayesephem=False, tnequad=False,
            be_type='orbel', is_wideband=False, use_dmdata=False, Tspan=None,
            select='backend', tm_marg=False, dense_like=False, tm_svd=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with only white and red noise:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Optional physical ephemeris modeling.


    :param psd:
        Choice of PSD function [e.g. powerlaw (default), turnover, tprocess]
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param Tspan: time baseline used to determine Fourier GP frequencies;
        derived from data if not specified
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(psd=psd, prior=amp_prior,
                         Tspan=Tspan, components=components)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2a(psrs, psd='powerlaw', noisedict=None, components=30,
             n_rnfreqs=None, n_gwbfreqs=None, gamma_common=None,
             delta_common=None, upper_limit=False, bayesephem=False,
             be_type='setIII', white_vary=False, is_wideband=False,
             use_dmdata=False, Tspan=None, select='backend', tnequad=False,
             pshift=False, pseed=None, psr_models=False,
             tm_marg=False, dense_like=False, tm_svd=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']

        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param Tspan: time baseline used to determine Fourier GP frequencies;
        derived from data if not specified
    :param psr_models:
        Return list of psr models rather than signal_base.PTA object.
    :param n_rnfreqs:
        Number of frequencies to use in achromatic rednoise model.
    :param n_gwbfreqs:
        Number of frequencies to use in the GWB model.
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    if n_gwbfreqs is None:
        n_gwbfreqs = components

    if n_rnfreqs is None:
        n_rnfreqs = components

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=n_rnfreqs)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=n_gwbfreqs, gamma_val=gamma_common,
                                delta_val=delta_common, name='gw',
                                pshift=pshift, pseed=pseed)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    if psr_models:
        return models
    else:
        # set up PTA
        if dense_like:
            pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
        else:
            pta = signal_base.PTA(models)

        # set white noise parameters
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

        return pta


def model_general(psrs, tm_var=False, tm_linear=False, tmparam_list=None,
                  tm_svd=False, tm_norm=True, noisedict=None, white_vary=False,
                  Tspan=None, modes=None, wgts=None, logfreq=False, nmodes_log=10,
                  common_psd='powerlaw', common_components=30, tnequad=False,
                  log10_A_common=None, gamma_common=None,
                  common_logmin=None, common_logmax=None,
                  orf='crn', orf_names=None, orf_ifreq=0, leg_lmax=5,
                  upper_limit_common=None, upper_limit=False,
                  red_var=True, red_psd='powerlaw', red_components=30, upper_limit_red=None,
                  red_select=None, red_breakflat=False, red_breakflat_fq=None,
                  bayesephem=False, be_type='setIII_1980', is_wideband=False, use_dmdata=False,
                  dm_var=False, dm_type='gp', dm_psd='powerlaw', dm_components=30,
                  upper_limit_dm=None, dm_annual=False, dm_chrom=False, dmchrom_psd='powerlaw',
                  dmchrom_idx=4, gequad=False, coefficients=False, pshift=False,
                  select='backend', tm_marg=False, dense_like=False,
                  delta_common=None):
    """
    Reads in list of enterprise Pulsar instances and returns a PTA
    object instantiated with user-supplied options.

    :param tm_var: boolean to vary timing model coefficients.
        [default = False]
    :param tm_linear: boolean to vary timing model under linear approximation.
        [default = False]
    :param tmparam_list: list of timing model parameters to vary.
        [default = None]
    :param tm_svd: stabilize timing model designmatrix with SVD.
        [default = False]
    :param tm_norm: normalize the timing model design matrix, or provide custom
        normalization. Alternative to 'tm_svd'.
        [default = True]
    :param noisedict: Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
        [default = None]
    :param white_vary: boolean for varying white noise or keeping fixed.
        [default = False]
    :param Tspan: timespan assumed for describing stochastic processes,
        in units of seconds. If None provided will find span of pulsars.
        [default = None]
    :param modes: list of frequencies on which to describe red processes.
        [default = None]
    :param wgts: sqrt summation weights for each frequency bin, i.e. sqrt(delta f).
        [default = None]
    :param logfreq: boolean for including log-spaced bins.
        [default = False]
    :param nmodes_log: number of log-spaced bins below 1/T.
        [default = 10]
    :param common_psd: psd of common process.
        ['powerlaw', 'spectrum', 'turnover', 'turnover_knee,', 'broken_powerlaw']
        [default = 'powerlaw']
    :param common_components: number of frequencies starting at 1/T for common process.
        [default = 30]
    :param log10_A_common: value of fixed log10_A_common parameter for
        fixed amplitude analyses.
        [default = None]
    :param gamma_common: fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
        [default = None]
    :param common_logmin: specify lower prior for common psd. This is a prior on log10_rho
        if common_psd is 'spectrum', else it is a prior on log amplitude
    :param common_logmax: specify upper prior for common psd. This is a prior on log10_rho
        if common_psd is 'spectrum', else it is a prior on log amplitude
    :param orf: comma de-limited string of multiple common processes with different orfs.
        [default = crn]
    :param orf_names: comma de-limited string of process names for different orfs. Manual
        control of these names is useful for embedding model_general within a hypermodel
        analysis for a process with and without hd correlations where we want to avoid
        parameter duplication.
        [default = None]
    :param orf_ifreq:
        Frequency bin at which to start the Hellings & Downs function with
        numbering beginning at 0. Currently only works with freq_hd orf.
        [default = 0]
    :param leg_lmax:
        Maximum multipole of a Legendre polynomial series representation
        of the overlap reduction function.
        [default = 5]
    :param upper_limit_common: perform upper limit on common red noise amplitude. Note
        that when perfoming upper limits it is recommended that the spectral index also
        be fixed to a specific value.
        [default = False]
    :param upper_limit: apply upper limit priors to all red processes.
        [default = False]
    :param red_var: boolean to switch on/off intrinsic red noise.
        [default = True]
    :param red_psd: psd of intrinsic red process.
        ['powerlaw', 'spectrum', 'turnover', 'tprocess', 'tprocess_adapt']
        [default = 'powerlaw']
    :param red_components: number of frequencies starting at 1/T for intrinsic red process.
        [default = 30]
    :param upper_limit_red: perform upper limit on intrinsic red noise amplitude. Note
        that when perfoming upper limits it is recommended that the spectral index also
        be fixed to a specific value.
        [default = False]
    :param red_select: selection properties for intrinsic red noise.
        ['backend', 'band', 'band+', None]
        [default = None]
    :param red_breakflat: break red noise spectrum and make flat above certain frequency.
        [default = False]
    :param red_breakflat_fq: break frequency for 'red_breakflat'.
        [default = None]
    :param bayesephem: boolean to include BayesEphem model.
        [default = False]
    :param be_type: flavor of bayesephem model based on how partials are computed.
        ['orbel', 'orbel-v2', 'setIII', 'setIII_1980']
        [default = 'setIII_1980']
    :param is_wideband: boolean for whether input TOAs are wideband TOAs. Will exclude
        ecorr from the white noise model.
        [default = False]
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if is_wideband.
        [default = False]
    :param dm_var: boolean for explicitly searching for DM variations.
        [default = False]
    :param dm_type: type of DM variations.
        ['gp', other choices selected with additional options; see below]
        [default = 'gp']
    :param dm_psd: psd of DM GP.
        ['powerlaw', 'spectrum', 'turnover', 'tprocess', 'tprocess_adapt']
        [default = 'powerlaw']
    :param dm_components: number of frequencies starting at 1/T for DM GP.
        [default = 30]
    :param upper_limit_dm: perform upper limit on DM GP. Note that when perfoming
        upper limits it is recommended that the spectral index also be
        fixed to a specific value.
        [default = False]
    :param dm_annual: boolean to search for an annual DM trend.
        [default = False]
    :param dm_chrom: boolean to search for a generic chromatic GP.
        [default = False]
    :param dmchrom_psd: psd of generic chromatic GP.
        ['powerlaw', 'spectrum', 'turnover']
        [default = 'powerlaw']
    :param dmchrom_idx: spectral index of generic chromatic GP.
        [default = 4]
    :param gequad: boolean to search for a global EQUAD.
        [default = False]
    :param coefficients: boolean to form full hierarchical PTA object;
        (no analytic latent-coefficient marginalization)
        [default = False]
    :param pshift: boolean to add random phase shift to red noise Fourier design
        matrices for false alarm rate studies.
        [default = False]
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    Default PTA object composition:
        1. fixed EFAC per backend/receiver system (per pulsar)
        2. fixed EQUAD per backend/receiver system (per pulsar)
        3. fixed ECORR per backend/receiver system (per pulsar)
        4. Red noise modeled as a power-law with 30 sampling frequencies
           (per pulsar)
        5. Linear timing model (per pulsar)
        6. Common-spectrum uncorrelated process modeled as a power-law with
           30 sampling frequencies. (global)
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'
    gp_priors = [upper_limit_red, upper_limit_dm, upper_limit_common]
    if all(ii is None for ii in gp_priors):
        amp_prior_red = amp_prior
        amp_prior_dm = amp_prior
        amp_prior_common = amp_prior
    else:
        amp_prior_red = 'uniform' if upper_limit_red else 'log-uniform'
        amp_prior_dm = 'uniform' if upper_limit_dm else 'log-uniform'
        amp_prior_common = 'uniform' if upper_limit_common else 'log-uniform'

    # timing model
    if not tm_var and not use_dmdata:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd, normed=tm_norm,
                                       coefficients=coefficients)
    elif not tm_var and use_dmdata:
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        # create new attribute for enterprise pulsar object
        for p in psrs:
            p.tmparams_orig = OrderedDict.fromkeys(p.t2pulsar.pars())
            for key in p.tmparams_orig:
                p.tmparams_orig[key] = (p.t2pulsar[key].val,
                                        p.t2pulsar[key].err)
        if not tm_linear:
            s = timing_block(tmparam_list=tmparam_list)
        else:
            pass

    # find the maximum time span to set GW frequency sampling
    if Tspan is not None:
        Tspan = Tspan
    else:
        Tspan = model_utils.get_tspan(psrs)

    if logfreq:
        fmin = 10.0
        modes, wgts = model_utils.linBinning(Tspan, nmodes_log,
                                             1.0 / fmin / Tspan,
                                             common_components, nmodes_log)
        wgts = wgts**2.0

    # red noise
    if red_var:
        s += red_noise_block(psd=red_psd, prior=amp_prior_red, Tspan=Tspan,
                             components=red_components, modes=modes, wgts=wgts,
                             coefficients=coefficients,
                             select=red_select, break_flat=red_breakflat,
                             break_flat_fq=red_breakflat_fq)

    # common red noise block
    crn = []
    if orf_names is None:
        orf_names = orf
    for elem, elem_name in zip(orf.split(','), orf_names.split(',')):
        if elem == 'zero_diag_bin_orf' or elem == 'zero_diag_legendre_orf':
            log10_A_val = log10_A_common
        else:
            log10_A_val = None
        crn.append(common_red_noise_block(psd=common_psd, prior=amp_prior_common, Tspan=Tspan,
                                          components=common_components,
                                          log10_A_val=log10_A_val, gamma_val=gamma_common,
                                          delta_val=None, orf=elem, name='gw_{}'.format(elem_name),
                                          orf_ifreq=orf_ifreq, leg_lmax=leg_lmax,
                                          coefficients=coefficients, pshift=pshift, pseed=None,
                                          logmin=common_logmin, logmax=common_logmax))
        # orf_ifreq only affects freq_hd model.
        # leg_lmax only affects (zero_diag_)legendre_orf model.
    crn = functools.reduce((lambda x, y: x+y), crn)
    s += crn

    # DM variations
    if dm_var:
        if dm_type == 'gp':
            s += dm_noise_block(gp_kernel='diag', psd=dm_psd,
                                prior=amp_prior_dm,
                                components=dm_components, gamma_val=None,
                                coefficients=coefficients)
        if dm_annual:
            s += dm_annual_signal()
        if dm_chrom:
            s += chromatic_noise_block(psd=dmchrom_psd, idx=dmchrom_idx,
                                       name='chromatic',
                                       components=dm_components,
                                       coefficients=coefficients)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []

    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            if gequad:
                s2 += white_signals.EquadNoise(log10_equad=parameter.Uniform(-8.5, -5),
                                               selection=selections.Selection(selections.no_selection),
                                               name='gequad')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / const.day
                tmax = p.toas.max() / const.day
                s3 = s2 + dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2,
                                                   sign=False, name='dmexp')
                models.append(s3(p))
            else:
                models.append(s2(p))
        else:
            s4 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            if gequad:
                s4 += white_signals.TNEquadNoise(log10_tnequad=parameter.Uniform(-8.5, -5),
                                                 selection=selections.Selection(selections.no_selection),
                                                 name='gequad')
            if '1713' in p.name and dm_var:
                tmin = p.toas.min() / const.day
                tmax = p.toas.max() / const.day
                s5 = s4 + dm_exponential_dip(tmin=tmin, tmax=tmax, idx=2,
                                                   sign=False, name='dmexp')
                models.append(s5(p))
            else:
                models.append(s4(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2b(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             bayesephem=False, be_type='orbel', is_wideband=False, components=30,
             use_dmdata=False, Tspan=None, select='backend', pshift=False, tnequad=False,
             tm_marg=False, dense_like=False, tm_svd=False, upper_limit=False,
             gamma_common=None):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2B from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Dipole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param Tspan: time baseline used to determine Fourier GP frequencies;
        derived from data if not specified
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole', pshift=pshift)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)
    # set white noise parameters

    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2c(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, Tspan=None, select='backend', tm_marg=False,
             dense_like=False, tm_svd=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2C from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Dipole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']

        2. Monopole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']

        3. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param Tspan: time baseline used to determine Fourier GP frequencies;
        derived from data if not specified
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole')

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='monopole', name='monopole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2d(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, n_rnfreqs=None, n_gwbfreqs=None,
             gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, Tspan=None, select='backend', pshift=False,
             tm_marg=False, dense_like=False, tm_svd=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2D from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. Monopole spatially correlated signal modeled with PSD.
        Default PSD is powerlaw. Available options
        ['powerlaw', 'turnover', 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param Tspan: time baseline used to determine Fourier GP frequencies;
        derived from data if not specified
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    if n_gwbfreqs is None:
        n_gwbfreqs = components

    if n_rnfreqs is None:
        n_rnfreqs = components

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=n_rnfreqs)

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=n_gwbfreqs, gamma_val=gamma_common,
                                orf='monopole', name='monopole', pshift=pshift)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_3a(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, n_rnfreqs=None, n_gwbfreqs=None,
             gamma_common=None, delta_common=None, upper_limit=False,
             bayesephem=False, be_type='setIII', is_wideband=False,
             use_dmdata=False, Tspan=None, select='backend',
             tnequad=False,
             pshift=False, pseed=None, psr_models=False,
             tm_marg=False, dense_like=False, tm_svd=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param delta_common:
        Fixed common red process spectral index value for higher frequencies in
        broken power law model.
        By default we vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param Tspan: time baseline used to determine Fourier GP frequencies;
        derived from data if not specified
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param psr_models:
        Return list of psr models rather than signal_base.PTA object.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    if n_gwbfreqs is None:
        n_gwbfreqs = components

    if n_rnfreqs is None:
        n_rnfreqs = components

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(psd='powerlaw',
                         prior=amp_prior,
                         Tspan=Tspan, components=n_rnfreqs)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=n_gwbfreqs, gamma_val=gamma_common,
                                delta_val=delta_common,
                                orf='hd', name='gw', pshift=pshift, pseed=pseed)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    if psr_models:
        return models
    else:
        # set up PTA
        if dense_like:
            pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
        else:
            pta = signal_base.PTA(models)

        # set white noise parameters
        if not white_vary or (is_wideband and use_dmdata):
            if noisedict is None:
                print('No noise dictionary provided!...')
            else:
                noisedict = noisedict
                pta.set_default_params(noisedict)

        return pta


def model_3b(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='setIII', is_wideband=False,
             use_dmdata=False, Tspan=None, select='backend', tm_marg=False,
             dense_like=False, tm_svd=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3B from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Dipole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        3. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param Tspan: time baseline used to determine Fourier GP frequencies;
        derived from data if not specified
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='hd', name='gw')

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_3c(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, Tspan=None, select='backend', tm_marg=False,
             dense_like=False, tm_svd=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3C from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Dipole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        3. Monopole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        4. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param Tspan: time baseline used to determine Fourier GP frequencies;
        derived from data if not specified
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    # timing model
    if is_wideband and use_dmdata:
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='hd', name='gw')

    # dipole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='dipole', name='dipole')

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='monopole', name='monopole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_3d(psrs, psd='powerlaw', noisedict=None, white_vary=False,
             components=30, gamma_common=None, upper_limit=False, tnequad=False,
             bayesephem=False, be_type='orbel', is_wideband=False,
             use_dmdata=False, Tspan=None, select='backend', tm_marg=False,
             dense_like=False, tm_svd=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 3D from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1. GWB with HD correlations modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Monopole signal modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        3. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum'] 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param Tspan: time baseline used to determine Fourier GP frequencies;
        derived from data if not specified
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    if Tspan is None:
        Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='hd', name='gw')

    # monopole
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                orf='monopole', name='monopole')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True,
                                                           model=be_type)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                       tnequad=tnequad, select=select)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False,
                                       tnequad=tnequad, select=select)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2a_drop_be(psrs, psd='powerlaw', noisedict=None, white_vary=False,
                     components=30, gamma_common=None, upper_limit=False,
                     is_wideband=False, use_dmdata=False, k_threshold=0.5,
                     pshift=False, tm_marg=False, dense_like=False, tm_svd=False,
                     tnequad=False,):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param k_threshold:
        Define threshold for dropout parameter 'k'.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                name='gw', pshift=pshift)

    # ephemeris model
    s += do.Dropout_PhysicalEphemerisSignal(use_epoch_toas=True,
                                            k_threshold=k_threshold)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True, tnequad=tnequad)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False, tnequad=tnequad)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_2a_drop_crn(psrs, psd='powerlaw', noisedict=None, white_vary=False,
                      components=30, gamma_common=None, upper_limit=False,
                      bayesephem=False, is_wideband=False, use_dmdata=False,
                      k_threshold=0.5, pshift=False, tm_marg=False,
                      dense_like=False, tm_svd=False, tnequad=False,):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    amp_name = '{}_log10_A'.format('gw')
    if amp_prior == 'uniform':
        log10_Agw = parameter.LinearExp(-18, -11)(amp_name)
    elif amp_prior == 'log-uniform' and gamma_common is not None:
        if np.abs(gamma_common - 4.33) < 0.1:
            log10_Agw = parameter.Uniform(-18, -14)(amp_name)
        else:
            log10_Agw = parameter.Uniform(-18, -11)(amp_name)
    else:
        log10_Agw = parameter.Uniform(-18, -11)(amp_name)

    gam_name = '{}_gamma'.format('gw')
    if gamma_common is not None:
        gamma_gw = parameter.Constant(gamma_common)(gam_name)
    else:
        gamma_gw = parameter.Uniform(0, 7)(gam_name)

    k_drop = parameter.Uniform(0.0, 1.0)  # per-pulsar

    drop_pl = do.dropout_powerlaw(log10_A=log10_Agw, gamma=gamma_gw,
                                  k_drop=k_drop, k_threshold=k_threshold)
    crn = gp_signals.FourierBasisGP(drop_pl, components=components,
                                    Tspan=Tspan, name='gw', pshift=pshift)
    s += crn

    # ephemeris model
    s += do.Dropout_PhysicalEphemerisSignal(use_epoch_toas=True)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True, tnequad=tnequad)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False, tnequad=tnequad)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


# Does not yet work with IPTA datasets due to white-noise modeling issues.
def model_chromatic(psrs, psd='powerlaw', noisedict=None, white_vary=False,
                    components=30, gamma_common=None, upper_limit=False,
                    bayesephem=False, is_wideband=False, use_dmdata=False,
                    pshift=False, idx=4, chromatic_psd='powerlaw',
                    c_psrs=['J1713+0747'], tm_marg=False, dense_like=False,
                    tm_svd=False, tnequad=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper + additional
    chromatic noise for given pulsars

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
        6. Chromatic noise for given pulsar list

    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param idx:
        Index of chromatic process (i.e DM is 2, scattering would be 4). If
        set to `vary` then will vary from 0 - 6 (This will be VERY slow!)
    :param chromatic_psd:
        PSD to use for chromatic noise. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param c_psrs:
        List of pulsars to use chromatic noise. 'all' will use all pulsars
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # white noise
    s += white_noise_block(vary=white_vary, inc_ecorr=not is_wideband,
                           tnequad=tnequad)

    # red noise
    s += red_noise_block(prior=amp_prior, Tspan=Tspan, components=components)

    # common red noise block
    s += common_red_noise_block(psd=psd, prior=amp_prior, Tspan=Tspan,
                                components=components, gamma_val=gamma_common,
                                name='gw', pshift=pshift)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # chromatic noise
    sc = chromatic_noise_block(psd=chromatic_psd, idx=idx)
    if c_psrs == 'all':
        s += sc
        models = [s(psr) for psr in psrs]
    elif len(c_psrs) > 0:
        models = []
        for psr in psrs:
            if psr.name in c_psrs:
                print('Adding chromatic model to PSR {}'.format(psr.name))
                snew = s + sc
                models.append(snew(psr))
            else:
                models.append(s(psr))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def model_bwm(psrs, likelihood=LogLikelihood, lookupdir=None, noisedict=None, tm_svd=False,
              Tmin_bwm=None, Tmax_bwm=None, skyloc=None, logmin=None, logmax=None,
              burst_logmin=-17, burst_logmax=-12, red_psd='powerlaw', components=30,
              dm_var=False, dm_psd='powerlaw', dm_annual=False, tnequad=False,
              upper_limit=False, bayesephem=False, wideband=False, tm_marg=False, dense_like=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with BWM model:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system (if NG channelized)
        4. Red noise modeled by a specified psd
        5. Linear timing model.
        6. Optional DM-variation modeling
    global:
        1. Deterministic GW burst with memory signal.
        2. Optional physical ephemeris modeling.

    :param psrs:
        list of enterprise.Pulsar objects for PTA
    :param noisedict:
        Dictionary of pulsar noise properties for fixed white noise.
        Can provide manually, or the code will attempt to find it.
    :param tm_svd:
        boolean for svd-stabilised timing model design matrix
    :param Tmin_bwm:
        Min time to search for BWM (MJD). If omitted, uses first TOA.
    :param Tmax_bwm:
        Max time to search for BWM (MJD). If omitted, uses last TOA.
    :param skyloc:
        Fixed sky location of BWM signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param logmin:
        Lower bound on log10_A of the red noise process in each pulsar`
    :param logmax:
        Upper bound on log10_A of the red noise process in each pulsar
    :param burst_logmin:
        Lower bound on the log10_A of the burst amplitude in each pulsar
    :param burst_logmax:
        Upper boudn on the log10_A of the burst amplitude in each pulsar
    :param red_psd:
        PSD to use for per pulsar red noise. Available options
        are ['powerlaw', 'turnover', tprocess, 'spectrum'].
    :param components:
        number of modes in Fourier domain processes (red noise, DM
        variations, etc)
    :param dm_var:
        include gaussian process DM variations
    :param dm_psd:
        power-spectral density for gp DM variations
    :param dm_annual:
        include a yearly period DM variation
    :param upper_limit:
        Perform upper limit on BWM amplitude. By default this is
        set to False for a 'detection' run.
    :param bayesephem:
        Include BayesEphem model.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    :return: instantiated enterprise.PTA object
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    if Tmin_bwm is None:
        Tmin_bwm = tmin/const.day
    if Tmax_bwm is None:
        Tmax_bwm = tmax/const.day

    if tm_marg:
        s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
    else:
        s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, psd=red_psd, Tspan=Tspan, components=components, logmin=logmin, logmax=logmax)

    # DM variations
    if dm_var:
        s += dm_noise_block(psd=dm_psd, prior=amp_prior, components=components,
                            gamma_val=None)
        if dm_annual:
            s += dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = dm_exponential_dip(tmin=54500, tmax=54900)

    # GW BWM signal block
    s += bwm_block(Tmin_bwm, Tmax_bwm, logmin=burst_logmin, logmax=burst_logmax,
                   amp_prior=amp_prior,
                   skyloc=skyloc, name='bwm')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True, tnequad=tnequad)
            if dm_var and 'J1713+0747' == p.name:
                s2 += dmexp
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False, tnequad=tnequad)
            if dm_var and 'J1713+0747' == p.name:
                s3 += dmexp
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_bwm_sglpsr(psr, likelihood=LogLikelihood, lookupdir=None,
                     noisedict=None, tm_svd=False, tnequad=False,
                     Tmin_bwm=None, Tmax_bwm=None,
                     burst_logmin=-17, burst_logmax=-12, fixed_sign=None,
                     red_psd='powerlaw', logmin=None,
                     logmax=None, components=30,
                     dm_var=False, dm_psd='powerlaw', dm_annual=False,
                     upper_limit=False, bayesephem=False,
                     wideband=False, tm_marg=False, dense_like=False):
    """
    Burst-With-Memory model for single pulsar runs
    Because all of the geometric parameters (pulsar_position, source_position, gw_pol) are all degenerate with each other in a single pulsar BWM search,
    this model can only search over burst epoch and residual-space ramp amplitude (t0, ramp_amplitude)

    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with single-pulsar BWM model (called a ramp):

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system (if NG channelized)
        4. Red noise modeled by a specified psd
        5. Linear timing model.
        6. Optional DM-variation modeling
        7. Deterministic GW burst with memory signal for this pulsar

    :param psr:
        enterprise.Pulsar objects for PTA. This model is only for one pulsar at a time.
    :param likelihood:
        The likelihood function to use. The options are [enterprise.signals.signal_base.LogLikelihood, enterprise.signals.signal_base.LookupLikelihood]
    :param noisedict:
        Dictionary of pulsar noise properties for fixed white noise.
        Can provide manually, or the code will attempt to find it.
    :param tm_svd:
        boolean for svd-stabilised timing model design matrix
    :param Tmin_bwm:
        Min time to search for BWM (MJD). If omitted, uses first TOA.
    :param Tmax_bwm:
        Max time to search for BWM (MJD). If omitted, uses last TOA.
    :param red_psd:
        PSD to use for per pulsar red noise. Available options
        are ['powerlaw', 'turnover', tprocess, 'spectrum'].
    :param components:
        number of modes in Fourier domain processes (red noise, DM
        variations, etc)
    :param dm_var:
        include gaussian process DM variations
    :param dm_psd:
        power-spectral density for gp DM variations
    :param dm_annual:
        include a yearly period DM variation
    :param upper_limit:
        Perform upper limit on BWM amplitude. By default this is
        set to False for a 'detection' run.
    :param bayesephem:
        Include BayesEphem model.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    :return: instantiated enterprise.PTA object


    """
    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set frequency sampling
    tmin = psr.toas.min()
    tmax = psr.toas.max()
    Tspan = tmax - tmin

    if Tmin_bwm is None:
        Tmin_bwm = tmin/const.day
    if Tmax_bwm is None:
        Tmax_bwm = tmax/const.day

    if tm_marg:
        s = gp_signals.MarginalizingTimingModel()
    else:
        s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, psd=red_psd, Tspan=Tspan, components=components, logmin=logmin, logmax=logmax)

    # DM variations
    if dm_var:
        s += dm_noise_block(psd=dm_psd, prior=amp_prior, components=components,
                            gamma_val=None)
        if dm_annual:
            s += dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = dm_exponential_dip(tmin=54500, tmax=54900)

    # GW BWM signal block
    s += bwm_sglpsr_block(Tmin_bwm, Tmax_bwm, amp_prior=amp_prior, name='ramp',
                          logmin=burst_logmin, logmax=burst_logmax, fixed_sign=fixed_sign)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # adding white-noise, and acting on psr objects
    models = []

    if 'NANOGrav' in psr.flags['pta'] and not wideband:
        s2 = s + white_noise_block(vary=False, inc_ecorr=True, tnequad=tnequad)
        if dm_var and 'J1713+0747' == psr.name:
            s2 += dmexp
        models.append(s2(psr))
    else:
        s3 = s + white_noise_block(vary=False, inc_ecorr=False, tnequad=tnequad)
        if dm_var and 'J1713+0747' == psr.name:
            s3 += dmexp
        models.append(s3(psr))

    # set up PTA
    # TODO: decide on a way to handle likelihood
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_fdm(psrs, noisedict=None, white_vary=False, tm_svd=False,
              Tmin_fdm=None, Tmax_fdm=None, gw_psd='powerlaw',
              red_psd='powerlaw', components=30, n_rnfreqs=None,
              n_gwbfreqs=None, gamma_common=None, delta_common=None,
              dm_var=False, dm_psd='powerlaw', dm_annual=False,
              upper_limit=False, bayesephem=False, wideband=False,
              pshift=False, pseed=None, model_CRN=False,
              amp_upper=-11, amp_lower=-18, tnequad=False,
              freq_upper=-7, freq_lower=-9,
              use_fixed_freq=False, fixed_freq=-8, tm_marg=False,
              dense_like=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with FDM model:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system (if NG channelized)
        4. Red noise modeled by a specified psd
        5. Linear timing model.
        6. Optional DM-variation modeling
        7. The pulsar phase term.
    global:
        1. Deterministic GW FDM signal.
        2. Optional physical ephemeris modeling.

    :param psrs:
        list of enterprise.Pulsar objects for PTA
    :param noisedict:
        Dictionary of pulsar noise properties for fixed white noise.
        Can provide manually, or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param tm_svd:
        boolean for svd-stabilised timing model design matrix
    :param Tmin_fdm:
        Min time to search for FDM (MJD). If omitted, uses first TOA.
    :param Tmax_fdm:
        Max time to search for FDM (MJD). If omitted, uses last TOA.
    :param gw_psd:
        PSD to use for the per pulsar GWB.
    :param red_psd:
        PSD to use for per pulsar red noise. Available options
        are ['powerlaw', 'turnover', tprocess, 'spectrum'].
    :param components:
        number of modes in Fourier domain processes (red noise, DM
        variations, etc)
    :param n_rnfreqs:
        Number of frequencies to use in achromatic rednoise model.
    :param n_gwbfreqs:
        Number of frequencies to use in the GWB model.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param dm_var:
        include gaussian process DM variations
    :param dm_psd:
        power-spectral density for gp DM variations
    :param dm_annual:
        include a yearly period DM variation
    :param upper_limit:
        Perform upper limit on FDM amplitude. By default this is
        set to False for a 'detection' run.
    :param bayesephem:
        Include BayesEphem model.
    :param wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param pshift:
        Option to use a random phase shift in design matrix. For testing the
        null hypothesis.
    :param pseed:
        Option to provide a seed for the random phase shift.
    :param model_CRN:
        Option to model the common red process in addition to the
        FDM signal.
    :param amp_upper, amp_lower, freq_upper, freq_lower:
        The log-space bounds on the amplitude and frequency priors.
    :param use_fixed_freq:
        Whether to do a fixed-frequency run and not search over the frequency.
    :param fixed_freq:
        The frequency value to do a fixed-frequency run with.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood

    :return: instantiated enterprise.PTA object
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    if n_gwbfreqs is None:
        n_gwbfreqs = components

    if n_rnfreqs is None:
        n_rnfreqs = components

    # find the maximum time span to set frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    if Tmin_fdm is None:
        Tmin_fdm = tmin/const.day
    if Tmax_fdm is None:
        Tmax_fdm = tmax/const.day

    # timing model
    if tm_marg:
        s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
    else:
        s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior, psd=red_psd, Tspan=Tspan, components=n_rnfreqs)

    # DM variations
    if dm_var:
        s += dm_noise_block(psd=dm_psd, prior=amp_prior, components=components,
                            gamma_val=None)
        if dm_annual:
            s += dm_annual_signal()

        # DM exponential dip for J1713's DM event
        dmexp = dm_exponential_dip(tmin=54500, tmax=54900)

    if model_CRN is True:
        # common red noise block
        s += common_red_noise_block(psd=gw_psd, prior=amp_prior, Tspan=Tspan,
                                    components=n_gwbfreqs, gamma_val=gamma_common,
                                    delta_val=delta_common, name='gw',
                                    pshift=pshift, pseed=pseed)

    # GW FDM signal block
    s += deterministic.fdm_block(Tmin_fdm, Tmax_fdm,
                                 amp_prior=amp_prior, name='fdm',
                                 amp_lower=amp_lower, amp_upper=amp_upper,
                                 freq_lower=freq_lower, freq_upper=freq_upper,
                                 use_fixed_freq=use_fixed_freq, fixed_freq=fixed_freq)

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not wideband:
            s2 = s + white_noise_block(vary=False, inc_ecorr=True, tnequad=tnequad)
            if dm_var and 'J1713+0747' == p.name:
                s2 += dmexp
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=False, inc_ecorr=False, tnequad=tnequad)
            if dm_var and 'J1713+0747' == p.name:
                s3 += dmexp
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if noisedict is None:
        print('No noise dictionary provided!...')
    else:
        noisedict = noisedict
        pta.set_default_params(noisedict)

    return pta


def model_cw(psrs, upper_limit=False, rn_psd='powerlaw', noisedict=None,
             white_vary=False, components=30, bayesephem=False, skyloc=None,
             log10_F=None, ecc=False, psrTerm=False, is_wideband=False,
             use_dmdata=False, gp_ecorr='basis_ecorr', tnequad=False,
             tm_marg=False, dense_like=False, tm_svd=False):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with CW model:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.
    global:
        1. Deterministic CW signal.
        2. Optional physical ephemeris modeling.

    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param rn_psd:
        psd to use in red_noise_block()
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param white_vary:
        boolean for varying white noise or keeping fixed.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param skyloc:
        Fixed sky location of CW signal search as [cos(theta), phi].
        Search over sky location if ``None`` given.
    :param log10_F:
        Fixed frequency of CW signal search.
        Search over frequency if ``None`` given.
    :param ecc:
        boolean or float
        if boolean: include/exclude eccentricity in search
        if float: use fixed eccentricity with eccentric model
    :psrTerm:
        boolean, include/exclude pulsar term in search
    :param is_wideband:
        Whether input TOAs are wideband TOAs; will exclude ecorr from the white
        noise model.
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
        is_wideband.
    :param tm_marg: Use marginalized timing model. In many cases this will speed
        up the likelihood calculation significantly.
    :param dense_like: Use dense or sparse functions to evalute lnlikelihood
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    """

    amp_prior = 'uniform' if upper_limit else 'log-uniform'

    # find the maximum time span to set GW frequency sampling
    tmin = np.min([p.toas.min() for p in psrs])
    tmax = np.max([p.toas.max() for p in psrs])
    Tspan = tmax - tmin

    # timing model
    if (is_wideband and use_dmdata):
        dmjump = parameter.Constant()
        if white_vary:
            dmefac = parameter.Uniform(pmin=0.1, pmax=10.0)
            log10_dmequad = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            dmefac = parameter.Constant()
            log10_dmequad = parameter.Constant()
            # dmjump = parameter.Constant()
        s = gp_signals.WidebandTimingModel(dmefac=dmefac,
                                           log10_dmequad=log10_dmequad, dmjump=dmjump,
                                           selection=selections.Selection(selections.by_backend),
                                           dmjump_selection=selections.Selection(selections.by_frontend))
    else:
        if tm_marg:
            s = gp_signals.MarginalizingTimingModel(use_svd=tm_svd)
        else:
            s = gp_signals.TimingModel(use_svd=tm_svd)

    # red noise
    s += red_noise_block(prior=amp_prior,
                         psd=rn_psd, Tspan=Tspan, components=components)

    # GW CW signal block
    if not ecc:
        s += deterministic.cw_block_circ(amp_prior=amp_prior,
                                         skyloc=skyloc,
                                         log10_fgw=log10_F,
                                         psrTerm=psrTerm, tref=tmin,
                                         name='cw')
    else:
        if type(ecc) is not float:
            ecc = None
        s += deterministic.cw_block_ecc(amp_prior=amp_prior,
                                        skyloc=skyloc, log10_F=log10_F,
                                        ecc=ecc, psrTerm=psrTerm,
                                        tref=tmin, name='cw')

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

    # adding white-noise, and acting on psr objects
    models = []
    for p in psrs:
        if 'NANOGrav' in p.flags['pta'] and not is_wideband:
            if gp_ecorr:
                s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True,
                                           gp_ecorr=True, name=gp_ecorr,
                                           tnequad=tnequad)
            else:
                s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True, tnequad=tnequad)
            models.append(s2(p))
        else:
            s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False, tnequad=tnequad)
            models.append(s3(p))

    # set up PTA
    if dense_like:
        pta = signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)
    else:
        pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print('No noise dictionary provided!...')
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta
