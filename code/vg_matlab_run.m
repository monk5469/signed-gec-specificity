function vg_matlab_run(subj, kname, coop, outfile, probe)
% Implementation check (Supplementary Note 13): calls the authors' unmodified fitting function (commit da592aa) on example
% participant subj (1..3) with configuration kname (K1..K4) and model coop (1 = cooperative-only, 0 = signed); saves GEC,
% FC_corr, FCsim, the stop line printed by the function and the elapsed time. rng fixed per (subj, K, model); probe = 1 times one simulation.
R = [getenv('SGEC_DATA_ROOT') 'raw/luppi_repo_da592aa/competitive-cooperative-hopf-da592aa/'];
addpath([R 'matlab/matlab_utils']); addpath([R 'matlab/common']);
maxNumCompThreads(1);
a = load([R 'data/matlab/gec_3/SC.mat']); b = load([R 'data/matlab/gec_3/FMRI.mat']);
SC = a.DTI_fibers_VolNorm_HCP{subj}; TS = b.BOLD_timeseries_HCP{subj};
c = struct('TR', 0.72, 'filter_low', 0.008, 'filter_high', 0.09, 'tau', 1, 'a', -0.02, 'max_iterations', 5000, ...
           'convergence_threshold', 0.001);
switch kname
    case 'K1', c.sigma = 0.01;  c.learning_rate_FC = 2e-4; c.learning_rate_FC_tau = 4e-5; c.max_C = 0.10; c.alphaL1 = 0;
    case 'K2', c.sigma = 0.01;  c.learning_rate_FC = 2e-4; c.learning_rate_FC_tau = 4e-5; c.max_C = 0.15; c.alphaL1 = 0;
    case 'K3', c.sigma = 0.001; c.learning_rate_FC = 1e-3; c.learning_rate_FC_tau = 1e-4; c.max_C = 0.15; c.alphaL1 = 1e-4;
    case 'K4', c.sigma = 0.001; c.learning_rate_FC = 1e-3; c.learning_rate_FC_tau = 1e-4; c.max_C = 0.15; c.alphaL1 = 0;
    otherwise, error('unknown configuration %s', kname);
end
kidx = str2double(kname(2));
rng(100000 + 1000 * subj + 10 * kidx + coop, 'twister');
if probe
    f = fcn_extract_frequencies({TS}, c.TR, c.filter_low, c.filter_high);
    t0 = tic; s = fcn_Hopf_simulate_BOLD_from_GEC(0.2 * SC ./ max(SC(:)), f, c.sigma, size(TS, 2), c.TR, c.a);
    fprintf('probe one simulation %.2f s, size %s\n', toc(t0), mat2str(size(s))); return
end
logf = [outfile '.log']; if exist(logf, 'file'), delete(logf); end
diary(logf); t0 = tic;
[GEC, FC_corr, FCsim] = fcn_Hopf_NonLinear_cooperative_competitive(TS, SC, c, logical(coop), 'onSC');
elapsed = toc(t0); diary off;
txt = fileread(logf); m = regexp(txt, '(Converged at iteration \d+|Error increased at iteration \d+)', 'match');
if isempty(m), stopline = 'max_iterations'; else, stopline = m{end}; end
save(outfile, 'GEC', 'FC_corr', 'FCsim', 'stopline', 'elapsed', 'subj', 'kname', 'coop', 'c');
fprintf('VG done %s subj %d coop %d FC_corr %.4f %s %.0f s\n', kname, subj, coop, FC_corr, stopline, elapsed);
end
