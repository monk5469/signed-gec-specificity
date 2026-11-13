function vj_matlab_run(subj, kname, coop, r, outfile)
% Bifurcation-parameter check (Supplementary Note 13): calls the authors' unmodified fitting function (commit da592aa) on
% example participant subj (1..3) with configuration kname (K1a, K3a, K4a, KR or K4r), model coop (1 = cooperative-only,
% 0 = signed) and random stream r (0..7); rng(400000 + 1000 subj + 100 kidx + 10 coop + r, 'twister'). Saves GEC, FC_corr,
% the stop line, the elapsed time and the regional frequencies of the authors' fcn_extract_frequencies.
R = [getenv('SGEC_DATA_ROOT') 'raw/luppi_repo_da592aa/competitive-cooperative-hopf-da592aa/'];
addpath([R 'matlab/matlab_utils']); addpath([R 'matlab/common']);
maxNumCompThreads(1);
a = load([R 'data/matlab/gec_3/SC.mat']); b = load([R 'data/matlab/gec_3/FMRI.mat']);
SC = a.DTI_fibers_VolNorm_HCP{subj}; TS = b.BOLD_timeseries_HCP{subj};
c = struct('TR', 0.72, 'filter_low', 0.008, 'filter_high', 0.09, 'tau', 1, 'max_iterations', 5000, 'convergence_threshold', 0.001);
switch kname
    case 'K1a', kidx = 1; c.a = -0.025; c.sigma = 0.01;  c.learning_rate_FC = 2e-4; c.learning_rate_FC_tau = 4e-5; c.max_C = 0.10; c.alphaL1 = 0;
    case 'K3a', kidx = 3; c.a = -0.025; c.sigma = 0.001; c.learning_rate_FC = 1e-3; c.learning_rate_FC_tau = 1e-4; c.max_C = 0.15; c.alphaL1 = 1e-4;
    case 'K4a', kidx = 4; c.a = -0.025; c.sigma = 0.001; c.learning_rate_FC = 1e-3; c.learning_rate_FC_tau = 1e-4; c.max_C = 0.15; c.alphaL1 = 0;
    case 'KR',  kidx = 5; c.a = -0.025; c.sigma = 0.005; c.learning_rate_FC = 1e-3; c.learning_rate_FC_tau = 1e-4; c.max_C = 0.15; c.alphaL1 = 1e-4;
    case 'K4r', kidx = 6; c.a = -0.02;  c.sigma = 0.001; c.learning_rate_FC = 1e-3; c.learning_rate_FC_tau = 1e-4; c.max_C = 0.15; c.alphaL1 = 0;
    otherwise, error('unknown configuration %s', kname);
end
freqs = fcn_extract_frequencies({TS}, c.TR, c.filter_low, c.filter_high);
rng(400000 + 1000 * subj + 100 * kidx + 10 * coop + r, 'twister');
logf = [outfile '.log']; if exist(logf, 'file'), delete(logf); end
diary(logf); t0 = tic;
[GEC, FC_corr, FCsim] = fcn_Hopf_NonLinear_cooperative_competitive(TS, SC, c, logical(coop), 'onSC');
elapsed = toc(t0); diary off;
txt = fileread(logf); m = regexp(txt, '(Converged at iteration \d+|Error increased at iteration \d+)', 'match');
if isempty(m), stopline = 'max_iterations'; else, stopline = m{end}; end
save(outfile, 'GEC', 'FC_corr', 'FCsim', 'stopline', 'elapsed', 'subj', 'kname', 'coop', 'r', 'c', 'freqs');
fprintf('VJ done %s subj %d coop %d r %d FC_corr %.4f %s %.0f s\n', kname, subj, coop, r, FC_corr, stopline, elapsed);
end
