
% HW9 Sample Script:  Hidden Markov Modeling of Character Sequences

% Add path to Kevin Murphy's HMM toolbox 
addpath(genpath('HMMall'));
% Download from: http://people.cs.ubc.ca/~murphyk/Software/HMM/hmm.html

% reset random number generator seed
reset(RandStream.getDefaultStream);

% load training & test datasets
trainText = readText('aliceTrain.txt');
testText  = readText('aliceTest.txt');
numChars  = 27;  % alphabet size: 26 letters plus spaces (underscore)

% convert datasets to numeric form
trainData = textToNum(trainText);
testData  = textToNum(testText);

%%
%% Part I:  EM Training & Model Order Selection
%%
load trained;

% train models of varying orders
M  = [1 5 10 15 20 30 40 50 60];
nM = length(M);

%{
T = 500;  % maximum number of EM iterations
trainLogLike = nan(T,nM);
priorProb = cell(1,nM);
transProb = cell(1,nM);
obsProb   = cell(1,nM);

for (kk = 1:nM) 
  fprintf(1, '\n*** Training with %d states: ***\n', M(kk));

  prior0 = normalise(rand(M(kk),1));
  trans0 = mk_stochastic(rand(M(kk),M(kk)));
  obs0   = mk_stochastic(rand(M(kk),numChars));

  [llTrace, priorProb{kk}, transProb{kk}, obsProb{kk}] = dhmm_em(trainData, ...
    prior0, trans0, obs0, 'max_iter', T, 'thresh', 1e-6, 'verbose', 1);
  trainLogLike(1:length(llTrace),kk) = llTrace;
end
%}
% compute AIC/BIC scores & test likelihoods for all models 
trainAIC = zeros(1,nM);
trainBIC = zeros(1,nM);
testLogLike = zeros(1,nM);

for (kk = 1:nM) 
  %TODO: Compute the number of free parameters, and the BIC score
  % d(M) = M*(W-1) = 26*M
  numParam = M(kk) -1 + M(kk)*(M(kk) - 1) + 26*M(kk);
  trainBIC(kk) = max(trainLogLike(:,kk)) - 0.5*numParam*log(length(trainData));
  trainAIC(kk) = max(trainLogLike(:,kk)) - numParam;

  %TODO: Compute the test data log-likelihood
  testLogLike(kk) = dhmm_logprob(testData, priorProb{kk}, transProb{kk}, obsProb{kk});
end
%TODO: Plot the AIC scores, BIC scores, and training likelihoods
figure;
subplot(2,2,1);
plot(trainAIC);
title('AIC');

subplot(2,2,2);
plot(trainBIC);
title('BIC');

subplot(2,2,3);
plot(max(trainLogLike));
title('Train Log Like');

subplot(2,2,4);
plot(testLogLike);
title('Test Log Like');
%TODO: Plot test likelihoods

% select best scoring models 
[valueBIC,bestBIC] = max(trainBIC);
[valueAIC,bestAIC] = max(trainAIC);
indTest = [1 bestBIC bestAIC nM];
numTest = length(indTest);

% simulate random text strings from trained models 
sampleText = cell(numTest,1);
for (kk = 1:numTest)
  sampleText{kk} = numToText(dhmm_sample(priorProb{indTest(kk)}, ...
    transProb{indTest(kk)}, obsProb{indTest(kk)}, 1, 500));
  sampleText{kk}
end

% 1:
% tnhdop_tteieuruhcrcwtaetnoisnh_i_loor_i_iertot__h_efoeegieehieb_ey_ia_wgob_ue_aat_q_ihtai_loi_n_dil_eeeret_ttltlr_t_hal___easuaylaseiho_mrert_se_l_rao_estian_wiusttnrynhtllsatnwiergdl_eedw_ndg_aagnthtraero__wfuts_p_arhueahenepbdisunerehewcputttdisi_dly_i_dinnkal__p_nr_unsd_eesondc_lan__onf_edh_hsh_se_desha_h_ni__h_h__oo_l_ckuueitsttahnibidejv__e_rr_atoi__otaooh_aoatonyee__ht__hbse___e_eatdetioonork_i_letce_rtbiidomeegchelaehsdelan___esayavid_wd_isr__njatgl_rrhhhtfrihhorda_ette_nthe_seseaeesetthr
% bestBIC:
% whe_poo_leghom_the_ot_yusle_at_alk_was_then_te_ling_mise_biwn_aisd_tise_as_larermoonet_sete_th_yume_thedr_hotho_ning_waared_holkw_o_nurgy_an_lir_is_tooo_gan_esew_than_ward_the_geis_fode_wade_moke_a_gothanelet_wonereed_ile_sird_twlin_osild_nushit_rerwond_haf_ane_gat_clacrrise_aiy_eine_ihan_ghes_it_mut_pony_salthavire_is_a_wene_hethaqopt_bot_wevd_ayoooot_cuueuyom_co_ainensg_sten_az_watte_sarftdoser_at_ats_ghed_armyo_gilg_tl_orl_th_puz_the_kbut_cterkup_ad_wor_ghe_moodeemus_hathe_wend_a_cu_mee_who_g
% bestAIC
% the_alise_abd_thad_theag_she_any_twaln_rer_sothe_nonid_whe_das_wee_loface_monrmold_sout_mas_vepe_fit_it_whatf_viso_mice_cathe_id_iy_ther_amit_wk_sourt_he_dou_he_mteanth_toe_sad_coo_louch_clans_lewemithe_this_wk_and_beave_ald_maitter_niyniml_huce_it_k_wonkofne_sor_the_beal_silnore_yteare_wumce_m_yazkennt_tiet_at_sats_abn_care_anieded_glat_hea_wat_ou_mecare_ser_imace_the_gantelt_ve_dust_as_itted_sou_thel_souked_gatt_i_sroclutwl_uttiny_ilg_ler_beled_ip_id_wanse_sor_tand_greareyt_a_fitto_she_how_tha
% nM
% shew_the_meack_the_yanne_said_the_mot_the_mou_onow_ane_ang_at_bes_fare_weat_the_sas_dfu_shens_ling_a_chxe_meto_wave_noretather_ack_alioule_sath_tuttey_tfels_al_tot_thisonouse_the_dar_dou_dousen_thang_ag_yotioostly_po_rary_bin_i_qeubbcly_cile_at_it_said_cut_ang_the_swillice_of_dorces_forco_buty_hiseat_hkidnts_it_ang_sare_fit_relp_intedile_hairy_go_sagter_sitter_preasgind_the_rour_sart_it_sean_i_powle_seaboupper_ra_at_sile_he_the_sattle_it_gones_thing_is_i_halole_er_yorce_btean_parve_cidenigo_sett

%%
%% Part II:  Filling in Deleted Letters
%%

% randomly remove subset of letters from test sentence
eraseRate = 0.2;
eraseInd  = rand(1,length(testText)) < eraseRate;
numErased = sum(eraseInd);

noisyText = testText;
noisyText(eraseInd) = '*';
noisyData = textToNum(noisyText);

% test each of the 4 models: single state, best BIC, best AIC, most states
denoisedText = cell(numTest,1);
denoisedData = cell(numTest,1);
numCorrected = zeros(numTest,1);

for (jj = 1:numTest) 
    
  kk = indTest(jj);
  disp(kk);
  %TODO: determine likelihood of observations given each hidden state
  newObsProb = zeros(size(obsProb{kk},1), size(obsProb{kk},2) + 1);
  newObsProb(:, 1:size(obsProb{kk},2)) = (1-eraseRate) .* obsProb{kk};
  newObsProb(:, end) = eraseRate;
  obslik = zeros(size(obsProb{kk},1), length(noisyData)); 
  for i=1:length(noisyData)
    obslik(:,i) = newObsProb(:, noisyData(i));
  end
  
  %TODO: compute posterior probability of hidden state sequence
  [alpha, beta, stateProb] = fwdback(priorProb{kk}, transProb{kk}, obslik);
  posProb = alpha .* beta; % p(x_t|z)
  
  % matrix of distribution of states for each point in data
  % find most likely value of each erased letter\
  % sum over x
  % p(y_t|z,z_t = *) = sum_x p(y_t | x_t) p(x_t | z)
  denoisedData{jj} = noisyData;
  
  for (cc = 1:length(noisyData)) 
    if (denoisedData{jj}(cc) == 28) 
      %TODO: Implement optimal estimator for missing letter      
      currProb = zeros(27,1);
      for i=1:27
         % for j=1:M(kk)
        currProb(i) = obsProb{kk}(:,i)'*posProb(:,cc);
        %  end
      end
      denoisedData{jj}(cc) = find(currProb == max(currProb));
    end
  end
  denoisedText{jj} = numToText(denoisedData{jj});

  % determine number of corrected erasures
  numCorrected(jj) = numErased - sum(denoisedData{jj} ~= testData);
end

% display denoised text for each model
noisyText(1:500)
for kk = 1:numTest
  denoisedText{kk}(1:500)
end

